# Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Workflows](#basic-workflows)
3. [Advanced Usage](#advanced-usage)
4. [Troubleshooting](#troubleshooting)

## Installation

```bash
cd vlm_refactoring
pip install -r requirements.txt
pip install -e .
```

## Basic Workflows

### Workflow 1: Model Inference

Run basic inference on a safety dataset:

```bash
python scripts/inference/run_inference.py \
    --model_name llava \
    --dataset StrongREJECT \
    --no_image \
    --output_dir ./result/inference
```

**Parameters:**
- `--model_name`: Model to use (llava, llava_next, intern, qwen, deepseek)
- `--dataset`: Dataset name
- `--no_image`: Text-only mode (omit for multimodal)
- `--image`: Image type (blank, noise, panda, etc.)
- `--output_dir`: Where to save results

### Workflow 2: PCA Analysis

Analyze hidden state structure:

```bash
# Analyze all layers
python scripts/analysis/run_pca.py \
    --model_name intern \
    --dataset StrongREJECT \
    --no_image \
    --layer_index all \
    --token_index -5 \
    --output_dir ./result/pca

# Analyze specific layer
python scripts/analysis/run_pca.py \
    --model_name intern \
    --dataset StrongREJECT \
    --no_image \
    --layer_index 20 \
    --token_index -5 \
    --output_dir ./result/pca
```

**Token indices by model:**
- LLaVA: `-2`
- LLaVA-NeXT: `-4`
- Qwen: `-2`
- InternVL: `-5`
- DeepSeek: `-1`

### Workflow 3: Distribution Analysis

Generate PC1 projection histograms:

```bash
python scripts/analysis/run_histogram.py \
    --model_name qwen \
    --dataset StrongREJECT \
    --no_image \
    --layer_index all \
    --token_index -2 \
    --output_dir ./result/histogram
```

### Workflow 4: Activation Steering

Modify model behavior via hooks:

```bash
# Make model safer
python scripts/inference/run_with_hook.py \
    --model_name llava \
    --dataset StrongREJECT \
    --anchor_dataset llmsafeguard \
    --hook_layer 22 \
    --hook_type safe \
    --alpha 1.0 \
    --no_image \
    --output_dir ./result/hook

# Make model less safe (for research)
python scripts/inference/run_with_hook.py \
    --model_name llava \
    --dataset XSTest \
    --anchor_dataset llmsafeguard \
    --hook_layer 22 \
    --hook_type unsafe \
    --alpha 1.0 \
    --no_image \
    --output_dir ./result/hook
```

## Advanced Usage

### Using the Library in Python

```python
import sys
sys.path.insert(0, '/path/to/vlm_refactoring')

from src.models import load_model
from src.datasets import load_dataset
from src.inference import generate_response
from src.analysis import extract_hidden_states, pca_basic
from src.inference.processor import build_prompt

# Load model
model, processor, tokenizer = load_model('llava')
model.eval()

# Load dataset
prompts, labels, imgs, _ = load_dataset('StrongREJECT', no_image=True)

# Generate response
response = generate_response(model, processor, 'llava', prompts[0], None)
print(response)

# Extract hidden states
hidden_states = extract_hidden_states(
    model, processor, 'llava', prompts[0], None, build_prompt
)

# Analyze with PCA
vectors = [h[:, -1, :].squeeze().cpu().numpy() for h in hidden_states]
eigenvalues, eigenvectors, variance_ratio, mean = pca_basic(vectors, top_k=5)
```

### Custom Hook Implementation

```python
from src.hooks import HookManager
import numpy as np

# Assume you have computed pc1_vectors for each layer
pc1_vectors = [...]  # Your PC1 vectors

# Create hook manager
manager = HookManager(
    model=model,
    all_layer_eigen_vecs=pc1_vectors,
    layer_indices=22,  # Layer to modify
    token_indices=2,   # Token position
    alpha=1.0,         # Strength
    max_uses=1         # Apply once per forward pass
)

# Generate with hook active
response = generate_response(model, processor, 'llava', prompt, None)

# Clean up
manager.remove()
```

### Batch Processing Multiple Datasets

```bash
#!/bin/bash
MODELS="llava llava_next intern qwen"
DATASETS="StrongREJECT AdvBench HarmBench XSTest"

for model in $MODELS; do
    for dataset in $DATASETS; do
        echo "Processing $model on $dataset"
        python scripts/inference/run_inference.py \
            --model_name $model \
            --dataset $dataset \
            --no_image \
            --output_dir ./result/batch/$model
    done
done
```

### Resolution Experiment (Image-based)

Test different image resolutions:

```python
import os
from PIL import Image

resolutions = [336, 672, 1008, 1344]
for res in resolutions:
    # Resize image
    img = Image.open('dataset/clean.jpeg').resize((res, res))
    img_path = f'dataset/temp_{res}.png'
    img.save(img_path)
    
    # Run inference
    os.system(f"""
        python scripts/inference/run_inference.py \
            --model_name llava_next \
            --dataset Figstep \
            --image {img_path} \
            --output_dir ./result/resolution
    """)
```

## Troubleshooting

### CUDA Out of Memory

```python
# Use smaller batch sizes or gradient checkpointing
import torch
torch.cuda.empty_cache()

# Or use CPU for analysis
device = 'cpu'
```

### Missing Model Paths

Edit `configs/models.yaml` to point to your model locations:

```yaml
models:
  llava:
    path: "/your/path/to/llava-1.5-13b-hf"
```

### Import Errors

Make sure you've installed in development mode:

```bash
pip install -e .
```

And that you're in the correct directory:

```bash
cd /path/to/vlm_refactoring
python scripts/...
```

### DeepSeek-specific Issues

The DeepSeek models require additional dependencies from `utils/model/DeepSeek_VL`. Ensure these are copied:

```bash
ls utils/model/DeepSeek_VL
ls utils/model/DeepSeek_VL2
```

## Tips & Best Practices

1. **Start small**: Test on a subset of data first
2. **Monitor GPU memory**: Use `nvidia-smi` to track usage
3. **Save intermediate results**: Hidden states can be reused
4. **Version control**: Track your experiment configs
5. **Document findings**: Keep notes on which layers/alphas work best

## Common Research Questions

**Q: Which layer should I use for hooks?**
A: Check `configs/models.yaml` for recommended layers per model, or run PCA analysis to find layers with high variance.

**Q: How do I choose alpha?**
A: Start with 0.5-1.0. Higher values = stronger effect. Use grid search if needed.

**Q: Can I combine multiple PC directions?**
A: Yes, you can linearly combine PC1, PC2, etc. or from different layers.

**Q: How to handle text+image experiments?**
A: Compare `--no_image` (text-only) vs `--image blank` vs `--image noise` to isolate effects.

