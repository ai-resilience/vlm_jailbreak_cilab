# Safety Analysis of Image Verification-Based Reasoning Methods in MLLMs

Recent advances in Multimodal Large Language Models (MLLMs) have introduced *image verification-based reasoning* methods that prompt models to re-attend to visual inputs during generation (e.g., "verify your response against the image"). While these approaches improve reasoning accuracy on mathematical and visual benchmarks, their implications for **safety** remain unexplored. 

This repository provides the evaluation framework for analyzing whether image verification prompts inadvertently weaken safety alignment—causing models to comply with harmful requests embedded in images (e.g., jailbreak attacks via typography or adversarial images). We evaluate both reasoning performance and safety across multiple MLLMs under default and lookback (image verification) prompting conditions, and further analyze attention distributions to understand the underlying mechanism.

---

## 1. Inference

We compare two prompting conditions:
- **Default**: Standard reasoning prompt without image verification instruction.
- **Lookback**: Includes an explicit image verification instruction (e.g., *"verify your response against the image during generation"*).

### Supported Models

| Model | HuggingFace ID |
|-------|----------------|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` |

### Datasets

| Category | Datasets | Source |
|----------|----------|--------|
| Reasoning | WeMath, MathVista, MathVerse, MathVision, GeoMath | HuggingFace |
| Safety | FigStep, MM-SafetyBench-Typo, MM-SafetyBench-SD, MM-SafetyBench-SD-TYPO | [FigStep](https://github.com/ThuCCSLab/FigStep), [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) |

### Data Preparation

Download the datasets and place them under `eval_safety/eval_data/`:

```
eval_safety/eval_data/
├── wemath/
├── mathvista/
├── mathverse/
├── mathvision/
├── GeoMath/
├── FigStep/
└── MM-SafetyBench/
```

Each dataset folder should contain the images and annotation files in their original format.

### Running Inference

```bash
# Default prompting
bash eval_safety/scripts/eval/Safety-default-inference.sh

# Lookback (image verification) prompting
bash eval_safety/scripts/eval/Safety-lookback-inference.sh
```

Before running, update the paths in the script:
```bash
EVAL_DIR="/absolute/path/to/eval_safety"
RESULTS_DIR="/path/to/results/..."
```

---

## 2. Evaluation

### Reasoning Accuracy

After inference, compute accuracy scores using:

```bash
python eval_safety/cal_score.py --folder_path /path/to/results
```

### Safety Evaluation

For safety evaluation (attack success rate), refer to:

```
vlm_jailbreak_cilab/src/evaluate
```

---

## 3. Analysis

### Attention Distribution

Visualize how attention to image tokens changes across generation steps:

```bash
bash eval_safety/scripts/attention_visualization/attention_distribution/safety-default-viz.sh
bash eval_safety/scripts/attention_visualization/attention_distribution/safety-lookback-viz.sh
```

### Attention Map

Generate spatial attention heatmaps overlaid on the input image:

```bash
bash eval_safety/scripts/attention_visualization/attention_map/safety-default-map.sh
bash eval_safety/scripts/attention_visualization/attention_map/safety-lookback-map.sh
```

## Acknowledgements

This evaluation framework builds upon the [Look-Back](https://arxiv.org/abs/2507.03019) project and evaluation tools from [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout).
