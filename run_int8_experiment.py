from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent
# Keep the recorded results and checked-in model files unchanged on reruns.
RESULTS_DIR = ROOT / 'rerun_output'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
BATCH_SIZES = [1, 32]
WARMUP_CALLS = 20
TIMING_ROUNDS = 10


def weight_payload_bytes(model_path: Path) -> int:
    """Count initializer bytes used as matrix weights, excluding biases/scales."""
    model = onnx.load(str(model_path), load_external_data=True)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    weight_names: set[str] = set()
    weight_input_positions = {
        'Gemm': (1,),
        'MatMul': (1,),
        'MatMulInteger': (1,),
        'DynamicQuantizeMatMul': (1,),
        'QLinearMatMul': (3,),
        'QLinearConv': (3,),
        'Conv': (1,),
    }
    for node in model.graph.node:
        for position in weight_input_positions.get(node.op_type, ()):
            if position < len(node.input) and node.input[position] in initializers:
                weight_names.add(node.input[position])
    if not weight_names:
        raise RuntimeError(
            f'Could not identify weight initializers in {model_path.name}; '
            'stopping so the weight-size field is not mislabeled.'
        )
    total = 0
    for name in weight_names:
        tensor = initializers[name]
        total += int(numpy_helper.to_array(tensor).nbytes)
    return total


def make_session(path: Path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=['CPUExecutionProvider']
    )
    return session, session.get_inputs()[0].name


def accuracy(session, input_name, X, y):
    output = session.run(None, {input_name: X.astype(np.float32, copy=False)})[0]
    predicted = np.asarray(output).reshape(-1)
    return float(np.mean(predicted == y) * 100.0)


def timed_batch(session, input_name, batch):
    start = time.perf_counter_ns()
    session.run(None, {input_name: batch})
    return (time.perf_counter_ns() - start) / 1_000_000.0


def percentile_iqr(values):
    q1, q3 = np.percentile(values, [25, 75], method='linear')
    return float(q3 - q1)


def main():
    dataset = load_digits()
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)
    splits = []

    for split_number, seed in enumerate(SEEDS, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            early_stopping=True,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        split_dir = RESULTS_DIR / f'split_{split_number}_seed_{seed}'
        split_dir.mkdir(parents=True, exist_ok=True)
        fp32_path = split_dir / 'classifier_fp32.onnx'
        int8_path = split_dir / 'classifier_int8.onnx'

        converted = convert_sklearn(
            model,
            initial_types=[('float_input', FloatTensorType([None, X.shape[1]]))],
        )
        fp32_path.write_bytes(converted.SerializeToString())
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=False,
        )

        fp32_session, fp32_input = make_session(fp32_path)
        int8_session, int8_input = make_session(int8_path)
        accuracy_fp32 = accuracy(fp32_session, fp32_input, X_test, y_test)
        accuracy_int8 = accuracy(int8_session, int8_input, X_test, y_test)

        batches = {size: X_test[:size].astype(np.float32, copy=False) for size in BATCH_SIZES}
        if any(len(batch) != size for size, batch in batches.items()):
            raise RuntimeError('The test split does not contain enough inputs for timing.')

        sessions = {
            'FP32': (fp32_session, fp32_input),
            'INT8': (int8_session, int8_input),
        }
        raw_rounds = {
            str(size): {'FP32': [], 'INT8': []}
            for size in BATCH_SIZES
        }
        for size, batch in batches.items():
            for session, input_name in sessions.values():
                for _ in range(WARMUP_CALLS):
                    session.run(None, {input_name: batch})
            for round_number in range(TIMING_ROUNDS):
                order = ['FP32', 'INT8'] if round_number % 2 == 0 else ['INT8', 'FP32']
                for label in order:
                    session, input_name = sessions[label]
                    raw_rounds[str(size)][label].append(
                        timed_batch(session, input_name, batch)
                    )

        latency = {}
        for size in BATCH_SIZES:
            latency[str(size)] = {}
            for label in ('FP32', 'INT8'):
                values = raw_rounds[str(size)][label]
                latency[str(size)][label] = {
                    'median_ms': float(np.median(values)),
                    'iqr_ms': percentile_iqr(values),
                    'rounds_ms': values,
                }

        split = {
            'split': split_number,
            'seed': seed,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'accuracy_fp32_pct': accuracy_fp32,
            'accuracy_int8_pct': accuracy_int8,
            'accuracy_difference_pp': accuracy_int8 - accuracy_fp32,
            'file_size_fp32_bytes': fp32_path.stat().st_size,
            'file_size_int8_bytes': int8_path.stat().st_size,
            'raw_weights_fp32_bytes': weight_payload_bytes(fp32_path),
            'raw_weights_int8_bytes': weight_payload_bytes(int8_path),
            'latency': latency,
            'training_iterations': int(model.n_iter_),
            'model_paths': {
                'fp32': str(fp32_path.relative_to(ROOT)),
                'int8': str(int8_path.relative_to(ROOT)),
            },
        }
        splits.append(split)
        print(json.dumps(split, indent=2))

    differences = [item['accuracy_difference_pp'] for item in splits]
    summary = {
        'status': 'completed',
        'date': time.strftime('%Y-%m-%d'),
        'machine': 'MacBook Air, Apple M5, macOS 27.0',
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'scikit_learn_version': sklearn.__version__,
        'onnxruntime_version': ort.__version__,
        'sklearn_onnx_version': skl2onnx_version,
        'onnx_version': onnx.__version__,
        'execution_provider': 'CPUExecutionProvider',
        'split_seeds': SEEDS,
        'warmup_calls_per_model_per_batch': WARMUP_CALLS,
        'timing_rounds_per_model_per_batch': TIMING_ROUNDS,
        'mean_accuracy_difference_pp': float(statistics.mean(differences)),
        'stdev_accuracy_difference_pp': float(statistics.stdev(differences)),
        'splits': splits,
    }
    out_path = RESULTS_DIR / 'results.json'
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
