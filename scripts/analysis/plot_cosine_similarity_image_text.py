#!/usr/bin/env python3
"""Plot cosine similarity between image-only and text-only hidden states across layers."""
import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import re

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity between image-only and text-only hidden states"
    )
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['Figstep', 'mm_typo'],
                       help='Dataset name')
    parser.add_argument('--token_idx', type=int, default=None,
                       choices=[-1, -2, -3, -4, -5],
                       help='Token index to analyze (required if pooling_mode=token)')
    parser.add_argument('--pooling_mode', type=str, default='token',
                       choices=['token', 'mean_pooling'],
                       help='Pooling mode: token (extract specific tokens) or mean_pooling (average across sequence)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Input directory (default: ../result/hidden_states)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: ../result/plots)')
    return parser.parse_args()


def get_resolutions(model_name: str) -> List[int]:
    """Get resolution list for each model."""
    if model_name == "intern":
        return [448, 896, 1344]
    elif model_name == "llava_next":
        return [336, 672]
    elif model_name == "deepseek2" or model_name == "deepseek":
        return [384, 768, 1152]
    elif model_name == "qwen" or model_name == "kimi":
        return [280, 560, 840, 1120, 1400, 1680]
    else:  # llava
        return [None]  # Use original resolution


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two tensors.
    
    Args:
        a: Tensor of shape [..., dim]
        b: Tensor of shape [..., dim] (broadcastable with a)
    
    Returns:
        Cosine similarity tensor of shape [...]
    """
    # Normalize
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    # Compute cosine similarity
    return (a_norm * b_norm).sum(dim=-1)


def load_hidden_states(file_path: str) -> torch.Tensor:
    """Load hidden states from .pt file.
    
    Args:
        file_path: Path to .pt file
    
    Returns:
        Tensor of shape [num_samples, num_layers, dim]
    """
    return torch.load(file_path, map_location='cpu')


def find_resolution_files(token_dir: str, model_name: str, dataset: str, pooling_mode: str = 'token') -> Dict[int, str]:
    """Find all resolution files in a token directory.
    
    Args:
        token_dir: Directory containing resolution files
        model_name: Model name
        dataset: Dataset name
        pooling_mode: Pooling mode ('token' or 'mean_pooling')
    
    Returns:
        Dictionary mapping resolution to file path
    """
    resolution_files = {}
    
    if not os.path.exists(token_dir):
        return resolution_files
    
    # Different patterns for different datasets and pooling modes
    if pooling_mode == 'mean_pooling':
        if dataset == 'Figstep':
            # Pattern: {model_name}_{dataset}_image_only_res_{resolution}_mean_pooling.pt
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_image_only_res_(\d+)_mean_pooling\.pt')
        elif dataset == 'mm_typo':
            # Pattern: {model_name}_{dataset}_res_{resolution}_mean_pooling.pt
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_res_(\d+)_mean_pooling\.pt')
        else:
            # Default pattern
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_res_(\d+)_mean_pooling\.pt')
    else:  # token mode
        if dataset == 'Figstep':
            # Pattern: {model_name}_{dataset}_image_only_res_{resolution}.pt
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_image_only_res_(\d+)\.pt')
        elif dataset == 'mm_typo':
            # Pattern: {model_name}_{dataset}_res_{resolution}.pt
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_res_(\d+)\.pt')
        else:
            # Default pattern
            pattern = re.compile(rf'{re.escape(model_name)}_{re.escape(dataset)}_res_(\d+)\.pt')
    
    for filename in os.listdir(token_dir):
        match = pattern.match(filename)
        if match:
            resolution = int(match.group(1))
            file_path = os.path.join(token_dir, filename)
            resolution_files[resolution] = file_path
    
    return resolution_files


def compute_layerwise_cosine_similarity(
    image_hidden: torch.Tensor,
    text_hidden: torch.Tensor
) -> np.ndarray:
    """Compute cosine similarity for each layer across samples.
    
    Args:
        image_hidden: Tensor of shape [num_samples, num_layers, dim]
        text_hidden: Tensor of shape [num_samples, num_layers, dim]
    
    Returns:
        Array of shape [num_layers] with mean cosine similarity per layer
    """
    num_samples, num_layers, dim = image_hidden.shape
    
    # Compute cosine similarity for each sample and layer
    # Shape: [num_samples, num_layers]
    similarities = cosine_similarity(image_hidden, text_hidden)
    
    # Average across samples for each layer
    # Shape: [num_layers]
    mean_similarities = similarities.mean(dim=0).numpy()
    
    return mean_similarities


def plot_cosine_similarity(
    similarities_by_res: Dict[int, np.ndarray],
    num_layers: int,
    model_name: str,
    dataset: str,
    token_idx: int = None,
    pooling_mode: str = 'token',
    output_path: str = None
):
    """Plot cosine similarity across layers for different resolutions.
    
    Args:
        similarities_by_res: Dictionary mapping resolution to similarity array [num_layers]
        num_layers: Number of layers
        model_name: Model name
        dataset: Dataset name
        token_idx: Token index (optional, used for title)
        pooling_mode: Pooling mode ('token' or 'mean_pooling')
        output_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Layer indices (0 to num_layers-1)
    layer_indices = np.arange(num_layers)
    
    # Plot each resolution
    for resolution in sorted(similarities_by_res.keys()):
        similarities = similarities_by_res[resolution]
        plt.plot(layer_indices, similarities, marker='o', label=f'Res {resolution}', linewidth=2, markersize=4)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    
    # Set title based on pooling mode
    if pooling_mode == 'mean_pooling':
        plt.title(f'Image-Text Cosine Similarity by Layer (Mean Pooling)\n{model_name} / {dataset}', fontsize=14)
    else:
        plt.title(f'Image-Text Cosine Similarity by Layer\n{model_name} / {dataset} / Token {token_idx}', fontsize=14)
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    # Validate arguments
    if args.pooling_mode == 'token' and args.token_idx is None:
        print("❌ Error: --token_idx is required when --pooling_mode=token")
        return
    
    # Set default directories
    if args.input_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.input_dir = str(project_root.parent / 'result' / 'hidden_states')
    
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'plots')
    
    # Construct paths
    model_dir = os.path.join(args.input_dir, args.model_name)
    
    # Paths for image-only and text-only (different for different datasets and pooling modes)
    if args.pooling_mode == 'mean_pooling':
        # mean_pooling mode: use mean_pooling subdirectory
        if args.dataset == 'Figstep':
            image_pooling_dir = os.path.join(model_dir, f"{args.dataset}_image_only", "mean_pooling")
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", "mean_pooling")
        elif args.dataset == 'mm_typo':
            image_pooling_dir = os.path.join(model_dir, args.dataset, "mean_pooling")
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", "mean_pooling")
        else:
            image_pooling_dir = os.path.join(model_dir, f"{args.dataset}_image_only", "mean_pooling")
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", "mean_pooling")
    else:
        # token mode: use token index subdirectory
        token_idx_str = str(args.token_idx)
        if args.dataset == 'Figstep':
            image_pooling_dir = os.path.join(model_dir, f"{args.dataset}_image_only", token_idx_str)
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", token_idx_str)
        elif args.dataset == 'mm_typo':
            image_pooling_dir = os.path.join(model_dir, args.dataset, token_idx_str)
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", token_idx_str)
        else:
            image_pooling_dir = os.path.join(model_dir, f"{args.dataset}_image_only", token_idx_str)
            text_pooling_dir = os.path.join(model_dir, f"{args.dataset}_text_only", token_idx_str)
    
    # Load text-only hidden states
    if args.pooling_mode == 'mean_pooling':
        text_file = os.path.join(text_pooling_dir, f"{args.model_name}_{args.dataset}_text_only_mean_pooling.pt")
    else:
        text_file = os.path.join(text_pooling_dir, f"{args.model_name}_{args.dataset}_text_only.pt")
    
    if not os.path.exists(text_file):
        print(f"❌ Text-only file not found: {text_file}")
        return
    
    print(f"Loading text-only hidden states from: {text_file}")
    text_hidden = load_hidden_states(text_file)
    print(f"   Shape: {text_hidden.shape} (samples, layers, dim)")
    
    # Find all resolution files for image-only
    resolution_files = find_resolution_files(image_pooling_dir, args.model_name, args.dataset, args.pooling_mode)
    
    if not resolution_files:
        print(f"❌ No image-only resolution files found in: {image_pooling_dir}")
        return
    
    print(f"\nFound {len(resolution_files)} resolution files:")
    for res in sorted(resolution_files.keys()):
        print(f"   Res {res}: {resolution_files[res]}")
    
    # Compute cosine similarity for each resolution
    similarities_by_res = {}
    num_layers = text_hidden.shape[1]
    
    for resolution, image_file in sorted(resolution_files.items()):
        print(f"\nProcessing resolution {resolution}...")
        print(f"   Loading: {image_file}")
        
        image_hidden = load_hidden_states(image_file)
        print(f"   Shape: {image_hidden.shape} (samples, layers, dim)")
        
        # Check shape compatibility
        if image_hidden.shape[0] != text_hidden.shape[0]:
            print(f"   ⚠️ Warning: Sample count mismatch (image: {image_hidden.shape[0]}, text: {text_hidden.shape[0]})")
            min_samples = min(image_hidden.shape[0], text_hidden.shape[0])
            image_hidden = image_hidden[:min_samples]
            text_hidden_aligned = text_hidden[:min_samples]
        else:
            text_hidden_aligned = text_hidden
        
        if image_hidden.shape[1] != text_hidden_aligned.shape[1]:
            print(f"   ❌ Error: Layer count mismatch (image: {image_hidden.shape[1]}, text: {text_hidden_aligned.shape[1]})")
            continue
        
        if image_hidden.shape[2] != text_hidden_aligned.shape[2]:
            print(f"   ❌ Error: Dimension mismatch (image: {image_hidden.shape[2]}, text: {text_hidden_aligned.shape[2]})")
            continue
        
        # Compute cosine similarity
        similarities = compute_layerwise_cosine_similarity(image_hidden, text_hidden_aligned)
        similarities_by_res[resolution] = similarities
        
        print(f"   ✅ Computed cosine similarity: shape {similarities.shape}, mean={similarities.mean():.4f}")
    
    # Plot results
    if similarities_by_res:
        if args.pooling_mode == 'mean_pooling':
            output_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_{args.dataset}_mean_pooling_cosine_similarity.png"
            )
        else:
            output_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_{args.dataset}_token{args.token_idx}_cosine_similarity.png"
            )
        
        plot_cosine_similarity(
            similarities_by_res,
            num_layers,
            args.model_name,
            args.dataset,
            args.token_idx,
            args.pooling_mode,
            output_path
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"   Processed {len(similarities_by_res)} resolutions")
        print(f"   Plot saved to: {output_path}")
    else:
        print("❌ No similarities computed. Check for errors above.")


if __name__ == '__main__':
    main()

