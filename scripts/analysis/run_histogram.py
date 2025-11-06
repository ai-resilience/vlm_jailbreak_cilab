#!/usr/bin/env python3
"""Generate histogram of PC1 projections."""
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.analysis import extract_hidden_states, extract_token_hidden_states, plot_histogram
from src.inference.processor import build_prompt
from src.models.base import find_norm, find_num_hidden_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PC1 projection histogram")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--no_image', action='store_true',
                       help='Use text-only mode')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use')
    parser.add_argument('--token_index', type=int, default=-1,
                       help='Token position to extract')
    parser.add_argument('--layer_index', type=str, default='all',
                       help='Layer index to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/histogram)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'histogram')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Get norm layer and num layers
    norm = find_norm(model)
    num_layers = find_num_hidden_layers(model)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    prompts, labels, imgs, _ = load_dataset(
        args.dataset,
        no_image=args.no_image,
        image=args.image
    )
    
    # Extract hidden states
    print(f"Extracting hidden states from {len(prompts)} samples...")
    vectors_all_layers = [[] for _ in range(num_layers)]
    
    for prompt, img in tqdm(zip(prompts, imgs), total=len(prompts)):
        hidden_states = extract_hidden_states(
            model, processor, args.model_name, prompt, img, build_prompt
        )
        
        vectors = extract_token_hidden_states(
            hidden_states, norm, args.token_index, num_layers
        )
        
        for idx in range(num_layers):
            vectors_all_layers[idx].append(vectors[idx])
    
    # Convert layer index
    if args.layer_index != 'all':
        layer_idx = int(args.layer_index)
    else:
        layer_idx = args.layer_index
    
    # Generate histogram
    print(f"Generating histogram for layer: {layer_idx}")
    modality = "text_only" if args.no_image else "image"
    save_path = os.path.join(
        args.output_dir,
        f"{args.model_name}_{modality}_{args.dataset}_histogram_{layer_idx}.png"
    )
    
    plot_histogram(vectors_all_layers, labels, layer_idx, save_path)
    
    print(f"✅ Histogram saved to: {save_path}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

