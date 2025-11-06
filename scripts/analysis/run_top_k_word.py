#!/usr/bin/env python3
"""Find top-k words using weight cosine similarity with PC1 or mean difference vectors."""
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.analysis import extract_hidden_states, extract_token_hidden_states, pca_basic, weight_cosine
from src.inference.processor import build_prompt
from src.models.base import find_norm, find_num_hidden_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Find top-k words using weight cosine similarity")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--method', type=str, choices=['pc1', 'mean_diff'], default='pc1',
                       help='Method to use: pc1 (PC1 vector) or mean_diff (mean difference by labels)')
    parser.add_argument('--no_image', action='store_true',
                       help='Use text-only mode')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use')
    parser.add_argument('--token_index', type=int, default=-1,
                       help='Token position to extract (-1 for last token)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of top words to return')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/top_k_word)')
    parser.add_argument('--layer_index', type=str, default='all',
                       help='Layer index to analyze ("all" or integer)')
    return parser.parse_args()


def compute_mean_difference_by_labels(vectors: List[np.ndarray], labels: List[int]) -> np.ndarray:
    """Compute mean difference vector between label groups.
    
    Args:
        vectors: List of vectors
        labels: Binary labels (0 or 1)
        
    Returns:
        Mean difference vector (mean_label1 - mean_label0)
    """
    labels_array = np.array(labels)
    vecs = np.stack(vectors, axis=0)
    
    # Compute mean for each label group
    label0_vecs = vecs[labels_array == 0]
    label1_vecs = vecs[labels_array == 1]
    
    if len(label0_vecs) == 0 or len(label1_vecs) == 0:
        raise ValueError("Both label groups must have at least one sample")
    
    mean0 = label0_vecs.mean(axis=0)
    mean1 = label1_vecs.mean(axis=0)
    
    # Return difference: label1_mean - label0_mean
    return mean1 - mean0


def main():
    args = parse_args()
    
    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'top_k_word')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Get norm layer and number of layers
    norm = find_norm(model)
    num_layers = find_num_hidden_layers(model)
    
    # Determine modality string
    modality = "text_only" if args.no_image else "image"
    
    # Load dataset and extract hidden states
    print(f"Loading dataset: {args.dataset}")
    prompts, labels, imgs, _ = load_dataset(
        args.dataset,
        no_image=args.no_image,
        image=args.image
    )
    
    # Validate labels for mean_diff method
    if args.method == 'mean_diff':
        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary (0 or 1) for mean_diff method. Found: {unique_labels}")
        if len(unique_labels) < 2:
            raise ValueError(f"Dataset must contain both label 0 and 1 for mean_diff method. Found: {unique_labels}")
        print(f"Label distribution: {sum(1 for l in labels if l == 0)} samples with label 0, {sum(1 for l in labels if l == 1)} samples with label 1")
    
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
    
    # Determine which layers to process
    if args.layer_index == 'all':
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(args.layer_index)]
    
    # Output file
    output_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_{modality}_{args.dataset}_{args.method}_top{args.top_k}.txt"
    )
    
    # Compute vectors and find top-k words
    print(f"Computing {args.method} vectors and finding top-{args.top_k} words...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for layer_idx in layer_indices:
            layer_vecs = vectors_all_layers[layer_idx]
            
            if args.method == 'pc1':
                # Compute PC1
                layer_vecs_array = np.stack(layer_vecs, axis=0)
                _, eigen_vecs, variance_ratio, _ = pca_basic(layer_vecs_array, top_k=10)
                pc1 = eigen_vecs[0]
                
                # Find top-k words for PC1+ and PC1-
                words_pos, scores_pos = weight_cosine(pc1, model, tokenizer, top_k=args.top_k)
                words_neg, scores_neg = weight_cosine(-pc1, model, tokenizer, top_k=args.top_k)
                
                # Write results
                f.write(f"==== Layer {layer_idx} ==== (Variance Ratio: {variance_ratio[0]:.4f})\n")
                f.write("PC1+\n")
                for token, score in zip(words_pos, scores_pos):
                    f.write(f"{token:>20} | {score.item():.4f}\n")
                f.write("\nPC1-\n")
                for token, score in zip(words_neg, scores_neg):
                    f.write(f"{token:>20} | {score.item():.4f}\n")
                f.write("\n\n")
                
            elif args.method == 'mean_diff':
                # Compute mean difference by labels
                delta_vec = compute_mean_difference_by_labels(layer_vecs, labels)
                
                # Find top-k words for delta+ and delta-
                words_pos, scores_pos = weight_cosine(delta_vec, model, tokenizer, top_k=args.top_k)
                words_neg, scores_neg = weight_cosine(-delta_vec, model, tokenizer, top_k=args.top_k)
                
                # Count samples per label
                label0_count = sum(1 for l in labels if l == 0)
                label1_count = sum(1 for l in labels if l == 1)
                
                # Write results
                f.write(f"==== Layer {layer_idx} ====\n")
                f.write(f"Mean Difference: Label 0 (n={label0_count}) -> Label 1 (n={label1_count})\n")
                f.write("Delta+ (Label1 - Label0)\n")
                for token, score in zip(words_pos, scores_pos):
                    f.write(f"{token:>20} | {score.item():.4f}\n")
                f.write("\nDelta- (Label0 - Label1)\n")
                for token, score in zip(words_neg, scores_neg):
                    f.write(f"{token:>20} | {score.item():.4f}\n")
                f.write("\n\n")
    
    print(f"✅ Results saved to: {output_file}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
