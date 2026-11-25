# INT8 Quantization for a Small Digit Classifier

This repository contains the supporting evidence for a project comparing a full-precision FP32 model with a dynamically quantized INT8 model for handwritten-digit classification. The project manuscript is submitted separately through Platon.

## Research question

How does dynamic INT8 quantization affect accuracy, model size, and inference latency for a small handwritten-digit classifier running on one CPU?

## Files

- `run_int8_experiment.py` — complete experiment script. It uses the digits data bundled with scikit-learn; no separate dataset download is needed.
- `results.json` — recorded measurements, including all ten timing rounds, software versions, and split details.
- `split_N_seed_S_classifier_fp32.onnx` and `split_N_seed_S_classifier_int8.onnx` — saved FP32 and INT8 models for each of five splits.
- `requirements.txt` and `PYTHON_VERSION.txt` — the package versions and Python version used for the recorded run.

## Method

The script loads the 1,797-example scikit-learn digits dataset, creates stratified 80/20 train/test splits with seeds 42 through 46, trains an MLP with 128 and 64 hidden units, exports FP32 ONNX models, and applies dynamic signed INT8 quantization. It records accuracy, file and raw weight sizes, and latency for batches of 1 and 32. Timing uses 20 warm-up calls and ten measured rounds.

## Recorded setup

The measurements were recorded on a MacBook Air with an Apple M5 CPU and macOS 27.0, using Python 3.12.14, scikit-learn 1.9.1, skl2onnx 1.20.0, ONNX 1.23.2, and ONNX Runtime 1.31.1 with its CPU provider.

## Re-run

Install the listed dependencies and run:

```bash
python -m pip install -r requirements.txt
python run_int8_experiment.py
```

New outputs are written to `rerun_output/`, so the recorded `results.json` and checked-in models remain unchanged. Timings can vary between runs and computers.

## Limits

The findings describe one small dataset, one MLP architecture, one Mac CPU, and one ONNX Runtime setup. They do not predict performance on other devices or models. The project did not measure energy use, execution-time memory, cold-start latency, or other CPUs. The exported model was quantized without applying ONNX Runtime's optional preprocessing recommendation.
