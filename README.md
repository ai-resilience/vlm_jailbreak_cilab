# VLM Safety Research Framework

A comprehensive framework for analyzing and manipulating Vision-Language Models (VLMs) to understand their safety mechanisms through representation engineering.

## 📋 Overview

This repository provides tools for:
- **Hidden State Analysis**: Extract and analyze internal representations from VLMs
- **PCA & Visualization**: Identify safety-relevant directions in activation space
- **Activation Steering**: Modify model behavior via targeted interventions
- **Safety Evaluation**: Comprehensive benchmarking on harmful/benign datasets

## 🏗️ Project Structure

```
vlm_refactoring/
├── src/                          # Core library
│   ├── models/                   # Model implementations
│   │   ├── base.py              # Base model interface
│   │   ├── llava.py             # LLaVA models
│   │   ├── qwen.py              # Qwen-VL model
│   │   ├── intern.py            # InternVL model
│   │   └── deepseek.py          # DeepSeek-VL models
│   ├── datasets/                 # Dataset loaders
│   │   ├── base.py
│   │   ├── figstep.py           # FigStep adversarial images
│   │   ├── strongreject.py      # StrongREJECT prompts
│   │   ├── xstest.py            # XSTest false positives
│   │   └── advbench.py          # AdvBench & HarmBench
│   ├── analysis/                 # Analysis tools
│   │   ├── pca.py               # PCA utilities
│   │   ├── hidden_states.py     # Hidden state extraction
│   │   └── visualization.py     # Plotting utilities
│   ├── hooks/                    # Activation steering
│   │   └── hook_manager.py      # Hook management
│   └── inference/                # Inference utilities
│       ├── processor.py         # Input processing
│       └── response.py          # Response generation
│
├── scripts/                      # Executable scripts
│   ├── inference/
│   │   ├── run_inference.py     # Basic inference
│   │   └── run_with_hook.py     # Inference with hooks
│   ├── analysis/
│   │   ├── run_pca.py           # PCA analysis
│   │   └── run_histogram.py     # Distribution analysis
│   └── eval/                     # Evaluation scripts
│
├── configs/                      # Configuration files
│   ├── models.yaml              # Model paths & settings
│   ├── datasets.yaml            # Dataset configurations
│   └── default.yaml             # Default parameters
│
├── dataset/                      # Data directory
├── eval/                         # Evaluation code
├── result/                       # Output directory
└── tests/                        # Unit tests

```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd vlm_refactoring

# Install dependencies
bash env_setting.sh

```

### Basic Usage

#### 1. Run Inference

```bash
python scripts/inference/run_inference.py \
    --model_name llava \
    --dataset StrongREJECT \
    --no_image \
    --output_dir ./result/inference
```

#### 2. Analyze Hidden States (PCA)

```bash
python scripts/analysis/run_pca.py \
    --model_name llava \
    --dataset StrongREJECT \
    --no_image \
    --layer_index all \
    --output_dir ./result/pca
```

#### 3. Generate PC1 Histograms

```bash
python scripts/analysis/run_histogram.py \
    --model_name llava \
    --dataset StrongREJECT \
    --no_image \
    --layer_index 22 \
    --output_dir ./result/histogram
```

#### 4. Run Inference with Activation Hooks

```bash
python scripts/inference/run_with_hook.py \
    --model_name llava \
    --dataset StrongREJECT \
    --anchor_dataset llmsafeguard \
    --hook_layer 22 \
    --hook_type safe \
    --alpha 1.0 \
    --no_image \
    --output_dir ./result/hook
```

## 🔬 Research Use Cases

### 1. Safety Direction Discovery

Find directions in activation space that correspond to safe/unsafe content:

```python
from src.models import load_model
from src.datasets import load_dataset
from src.analysis import extract_hidden_states, pca_basic

# Load model
model, processor, tokenizer = load_model('llava')

# Extract hidden states
prompts, labels, imgs, _ = load_dataset('StrongREJECT', no_image=True)
# ... extract and analyze
```

### 2. Activation Steering

Modify model behavior by injecting safety-relevant directions:

```python
from src.hooks import HookManager

# Create hook with PC1 direction
manager = HookManager(
    model, 
    all_layer_eigen_vecs,
    layer_indices=22,
    token_indices=2,
    alpha=1.0,
    max_uses=1
)

# Generate with modified activations
response = generate_response(model, processor, 'llava', prompt, img)
manager.remove()
```

### 3. Multi-Modal Safety Analysis

Analyze how different image types affect safety:

```bash
# Test with blank images
python scripts/inference/run_inference.py \
    --model_name intern \
    --dataset Figstep \
    --image blank

# Test with adversarial images
python scripts/inference/run_inference.py \
    --model_name intern \
    --dataset Figstep \
    --image noise
```

## 📊 Supported Models

| Model | Architecture | Size | Notes |
|-------|-------------|------|-------|
| LLaVA 1.5 | Vision Encoder + LLaMA | 13B | Base model |
| LLaVA-NeXT | Improved vision encoder | 7B | Better resolution |
| Qwen2.5-VL | Qwen architecture | 7B | Multilingual |
| InternVL3 | Hybrid architecture | 8B | State-of-the-art |
| DeepSeek-VL | Custom architecture | 7B | Specialized |

## 📚 Supported Datasets

### Safety Benchmarks
- **StrongREJECT**: Harmful prompt dataset
- **AdvBench**: Adversarial behaviors
- **HarmBench**: Standardized harmful prompts
- **SorryBench**: Comprehensive safety evaluation

### False Positive Tests
- **XSTest**: Benign prompts that trigger refusals

### Multimodal Attacks
- **FigStep**: Adversarial image jailbreaks

## 🔧 Advanced Configuration

Edit `configs/models.yaml` to customize model paths and parameters:

```yaml
models:
  llava:
    path: "/path/to/llava-1.5-13b-hf"
    token_index: 2
    key_layer: 22
```

Edit `configs/default.yaml` for experiment settings:

```yaml
inference:
  max_new_tokens: 128
  do_sample: false

hooks:
  alpha: 1.0
  max_uses: 1
```

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{vlm_safety_framework,
  title={VLM Safety Research Framework},
  year={2025},
  url={https://github.com/ai-resilience/vlm_jailbreak_cilab}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original codebase and research methodology
- HuggingFace Transformers team
- Open-source VLM model developers

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note**: This framework is for research purposes only. Be responsible when working with harmful content datasets.

