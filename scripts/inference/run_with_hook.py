#!/usr/bin/env python3
"""Run inference with activation hooks."""
import sys
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.inference import generate_response
from src.hooks import HookManager
from src.analysis import extract_hidden_states, extract_token_hidden_states, pca_basic


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM inference with hooks")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--anchor_dataset', type=str, required=True,
                       help='Anchor dataset for computing PC1')
    parser.add_argument('--hook_layer', type=int, required=True,
                       help='Layer index to inject hook')
    parser.add_argument('--hook_token', type=int, default=1,
                       help='Token position for hook')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Scaling factor for injection')
    parser.add_argument('--hook_type', type=str, choices=['safe', 'unsafe'], required=True,
                       help='Direction of injection')
    parser.add_argument('--no_image', action='store_true',
                       help='Use text-only mode')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use')
    parser.add_argument('--output_dir', type=str, default='./result/hook',
                       help='Output directory')
    return parser.parse_args()


def compute_pc1_directions(model, processor, model_name, anchor_dataset, hook_token, num_layers):
    """Compute PC1 directions from anchor dataset."""
    from src.models.base import BaseVLM
    from src.inference.processor import build_prompt
    
    # Get norm layer
    norm = None
    for path in ["language_model.norm", "language_model.model.norm"]:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            norm = obj
            break
        except AttributeError:
            continue
    
    # Load anchor data
    print(f"Loading anchor dataset: {anchor_dataset}")
    prompts, labels, imgs, _ = load_dataset(anchor_dataset, no_image=True)
    
    # Extract hidden states
    print("Extracting hidden states from anchor dataset...")
    vectors_all_layers = [[] for _ in range(num_layers)]
    
    for prompt, img in tqdm(zip(prompts, imgs), total=len(prompts)):
        hidden_states = extract_hidden_states(
            model, processor, model_name, prompt, img, build_prompt
        )
        
        for idx in range(num_layers):
            vec = hidden_states[idx][:, -hook_token, :]
            normed = norm(vec).squeeze(0).detach().cpu().float().numpy()
            vectors_all_layers[idx].append(normed)
    
    # Compute PC1 for each layer
    print("Computing PC1 directions...")
    all_layer_eigen_vecs = []
    for layer_idx in range(num_layers):
        layer_vecs = vectors_all_layers[layer_idx]
        _, eigen_vecs, _, pca_mean = pca_basic(layer_vecs, top_k=10)
        pc1 = eigen_vecs[0]
        
        # Align PC1 to unsafe direction
        unsafe_mean = np.asarray(layer_vecs)[np.asarray(labels) == 0].mean(axis=0)
        if (unsafe_mean - pca_mean) @ eigen_vecs[0] < 0:
            pc1 = -eigen_vecs[0]
        
        all_layer_eigen_vecs.append(pc1)
    
    return all_layer_eigen_vecs, labels


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Get number of layers
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
    elif hasattr(model.config, 'llm_config'):
        num_layers = model.config.llm_config.num_hidden_layers
    else:
        num_layers = model.config.language_config.num_hidden_layers
    
    # Compute PC1 directions
    all_layer_eigen_vecs, anchor_labels = compute_pc1_directions(
        model, processor, args.model_name,
        args.anchor_dataset, args.hook_token, num_layers
    )
    
    # Flip direction if safe hook
    if args.hook_type == 'safe':
        all_layer_eigen_vecs = [-pc1 for pc1 in all_layer_eigen_vecs]
    
    # Load target dataset
    print(f"Loading target dataset: {args.dataset}")
    prompts, labels, imgs, _ = load_dataset(
        args.dataset,
        no_image=args.no_image,
        image=args.image
    )
    
    # Output file
    modality = "text_only" if args.no_image else "image"
    output_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_{modality}_{args.dataset}_{args.hook_type}_hook_L{args.hook_layer}.jsonl"
    )
    
    # Run inference with hooks
    print(f"Running inference with {args.hook_type} hook on layer {args.hook_layer}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, img in tqdm(zip(prompts, imgs), total=len(prompts)):
            # Create hook manager
            manager = HookManager(
                model, all_layer_eigen_vecs,
                layer_indices=args.hook_layer,
                token_indices=args.hook_token,
                alpha=args.alpha,
                max_uses=1
            )
            
            # Generate with hook
            response = generate_response(model, processor, args.model_name, prompt, img)
            
            # Remove hook
            manager.remove()
            
            result = {
                'prompt': prompt,
                'response': response
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"✅ Results saved to: {output_file}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

