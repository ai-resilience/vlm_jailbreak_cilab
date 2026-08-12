# Multimodal Input Order and LVLM Safety

[한국어 README](README_KO.md)

This repository reproduces the study of how image–text input order affects safety alignment in large vision-language models. The same multimodal request is evaluated with the image before the text (**Image First**) and with the text before the image (**Text First**). Experiments cover SafeBench (Typo), MM-SafetyBench (SD+Typo), and MM-SafetyBench (SD) using 500 examples per benchmark.

The evaluated models are InternVL3-8B, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct, and Qwen3-VL-8B-Thinking. Attack success rate (ASR) is measured with Refusal, Target String, LlamaGuard3-8B, WildGuard, and the paper's ensemble method (EM).

## Setup

```bash
cd experiments/input_order_effect
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Model weights and benchmark images are not included. See [DATASETS.md](docs/DATASETS.md) for dataset preparation and licensing notes. Configure local paths in `configs/experiment.json`; all paths may be relative to the repository root.

## Reproduction workflow

```bash
# 1. Build deterministic 500-example manifests
python scripts/prepare_data.py --config configs/experiment.json

# 2. Run one benchmark/model pair (repeat or use your scheduler)
python scripts/run_inference.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 3. Extract last-input-token activations
python scripts/extract_activations.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 4. Evaluate responses
python scripts/evaluate_responses.py --config configs/experiment.json \
  --benchmark safebench_typo --model qwen25_vl_7b_instruct

# 5. Reproduce paper tables and figures
python scripts/generate_tables.py --results-dir results/evaluations
python scripts/generate_figures.py --config configs/experiment.json
```

Output files use the condition names `image_first.json` and `text_first.json`. The EM score follows the paper and is the per-example majority vote over Refusal, Target String, and LlamaGuard.

## Paper figures

- `figure2_wildguard_asr`: WildGuard ASR by model and input order.
- `figure3_layer_cosine`: layer-wise cosine similarity to the opposite refusal direction.
- `figure4_pca`: 2D PCA of harmless, harmful, and attack representations.

Run `python scripts/pilot_test.py` for a lightweight test that does not download models or datasets.
