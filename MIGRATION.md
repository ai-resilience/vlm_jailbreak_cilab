# Migration Guide: vlm_copy_251104 → vlm_refactoring

## Overview

This guide helps you migrate from the old codebase structure to the new refactored framework.

## Key Changes

### Directory Structure

**Old:**
```
vlm_copy_251104/
├── main.py
├── inference_fig.py
├── inference_res.py
├── histogram.py
├── utils/
│   ├── load_model.py
│   ├── load_dataset.py
│   ├── get_hidden.py
│   ├── get_response.py
│   ├── get_pca.py
│   └── load_hook.py
├── dataset/
└── eval/
```

**New:**
```
vlm_refactoring/
├── src/
│   ├── models/
│   ├── datasets/
│   ├── analysis/
│   ├── hooks/
│   └── inference/
├── scripts/
│   ├── inference/
│   ├── analysis/
│   └── eval/
├── configs/
├── dataset/ (symlink)
└── eval/ (symlink)
```

## Code Migration Examples

### 1. Loading Models

**Old:**
```python
from utils.load_model import load_model
model, processor, tokenizer = load_model("llava")
```

**New:**
```python
from src.models import load_model
model, processor, tokenizer = load_model("llava")
```

### 2. Loading Datasets

**Old:**
```python
from utils.load_dataset import load_dataset
prompts, labels, imgs, types = load_dataset(name="StrongREJECT", no_image=False, image="blank")
```

**New:**
```python
from src.datasets import load_dataset
prompts, labels, imgs, types = load_dataset("StrongREJECT", no_image=False, image="blank")
```

### 3. Getting Responses

**Old:**
```python
from utils.get_response import get_response
res = get_response(model, processor, model_name, prompt, img)
```

**New:**
```python
from src.inference import generate_response
res = generate_response(model, processor, model_name, prompt, img)
```

### 4. Extracting Hidden States

**Old:**
```python
from utils.get_hidden import get_hidden
vec = get_hidden(model, processor, model_name, prompt, img)
```

**New:**
```python
from src.analysis import extract_hidden_states
from src.inference.processor import build_prompt

hidden_states = extract_hidden_states(
    model, processor, model_name, prompt, img, build_prompt
)
```

### 5. PCA Analysis

**Old:**
```python
from utils.get_pca import pca_basic, pca_graph
eigenvalues, eigenvectors, variance_ratio, mean = pca_basic(vectors, top_k=5)
pca_graph(vectors, labels, pca_layer_index="all", save_path="./result/pca.png")
```

**New:**
```python
from src.analysis import pca_basic, pca_graph
eigenvalues, eigenvectors, variance_ratio, mean = pca_basic(vectors, top_k=5)
pca_graph(vectors, labels, "all", save_path="./result/pca.png")
```

### 6. Using Hooks

**Old:**
```python
from utils.load_hook import HookManager
manager = HookManager(model, all_layer_eigen_vecs, layer_indices, alpha, max_uses)
```

**New:**
```python
from src.hooks import HookManager
manager = HookManager(
    model, all_layer_eigen_vecs,
    layer_indices=layer_indices,
    token_indices=token_indices,
    alpha=alpha,
    max_uses=max_uses
)
```

## Script Migration

### Old main.py → New Scripts

**Old workflow (commented code in main.py):**
```python
# Inference
prompts, labels, imgs, _ = load_dataset(name="StrongREJECT", no_image=True)
for prompt, img in zip(prompts, imgs):
    res = get_response(model, processor, model_name, prompt, img)
    # save...
```

**New workflow:**
```bash
python scripts/inference/run_inference.py \
    --model_name llava \
    --dataset StrongREJECT \
    --no_image
```

### Old inference_fig.py → New Scripts

**Old:**
```bash
python inference_fig.py --model_name intern --dataset Figstep
```

**New:**
```bash
python scripts/inference/run_inference.py \
    --model_name intern \
    --dataset Figstep
```

### Old histogram.py → New Scripts

**Old:**
```bash
python histogram.py --model_name llava --dataset StrongREJECT
```

**New:**
```bash
python scripts/analysis/run_histogram.py \
    --model_name llava \
    --dataset StrongREJECT \
    --layer_index all
```

## Configuration Migration

Instead of hardcoding paths in code, use configuration files:

**configs/models.yaml:**
```yaml
models:
  llava:
    path: "/mnt/server11_hard4/kihyun/mil/llava-1.5-13b-hf"
    token_index: 2
    key_layer: 22
```

**configs/datasets.yaml:**
```yaml
datasets:
  StrongREJECT:
    huggingface_id: "walledai/StrongREJECT"
    type: "harmful"
```

## Backward Compatibility

The old codebase is preserved at `vlm_copy_251104/`. You can still run old scripts:

```bash
cd /mnt/server16_hard1/kihyun/vlm_copy_251104
python main.py
```

## Checklist for Migration

- [ ] Update import statements
- [ ] Update function calls (especially renamed functions)
- [ ] Move hardcoded configs to YAML files
- [ ] Test with small dataset first
- [ ] Verify output paths
- [ ] Update any custom scripts

## Getting Help

If you encounter issues:
1. Check `USAGE.md` for correct syntax
2. Compare with examples in `scripts/`
3. Verify imports with `python -c "from src.models import load_model"`

