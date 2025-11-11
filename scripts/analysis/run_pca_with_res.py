#!/usr/bin/env python3
"""Run PCA analysis on hidden states with resolution adjustment."""
import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.analysis import extract_hidden_states, extract_token_hidden_states, pca_graph
from src.inference.processor import build_prompt
from src.models.base import find_norm, find_num_hidden_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA analysis with resolution adjustment")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use (blank, panda, noise, etc.)')
    parser.add_argument('--token_index', type=int, default=-1,
                       help='Token position to extract (-1 for last token)')
    parser.add_argument('--layer_index', type=str, default='all',
                       help='Layer index to visualize ("all" or integer)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/pca)')
    return parser.parse_args()


def get_resolutions(model_name: str) -> list:
    """Get resolution list for each model (preserve aspect ratio)."""
    if model_name == "intern":
        return [448, 896, 1344]
    elif model_name == "llava_next":
        return [336, 672]
    elif model_name == "deepseek2" or model_name == "deepseek":
        return [384, 768, 1152]
    elif model_name == "qwen" or model_name == "kimi":
        return [560, 1680]
        # return [280, 560, 840, 1120, 1400, 1680]
    else:  # llava
        return [None]  # Use original resolution


def load_refusal_labels(model_name: str, dataset: str, resolution: int, project_root: Path) -> list:
    """Load refusal labels from evaluation result file.
    
    Args:
        model_name: Model name
        dataset: Dataset name
        resolution: Resolution value
        project_root: Project root path
        
    Returns:
        List of refusal labels (0 for False, 1 for True), or None if file not found
    """
    resolution_str = str(resolution) if resolution else "original"
    eval_file = project_root.parent / 'result' / 'evaluation' / model_name / dataset / \
                f"{model_name}_image_{dataset}_response_{resolution_str}_keyword.jsonl"
    
    if not eval_file.exists():
        print(f"⚠️  Evaluation file not found: {eval_file}")
        return None
    
    refusal_labels = []
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    refusal = data.get("refusal", "False")
                    # Convert "True"/"False" string to 1/0
                    refusal_labels.append(1 if refusal == "True" else 0)
        print(f"✅ Loaded {len(refusal_labels)} refusal labels from evaluation file")
        return refusal_labels
    except Exception as e:
        print(f"⚠️  Error loading refusal labels: {e}")
        return None


def main():
    args = parse_args()
    
    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'pca')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if dataset has its own images
    datasets_with_images = ["mm_text", "mm_typo", "mm_sd_typo", "Figstep"]
    has_dataset_images = args.dataset in datasets_with_images
    
    # Always use image mode (resolution adjustment requires images)
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    prompts, labels, imgs, _ = load_dataset(
        args.dataset,
        no_image=False,  # Always use images for resolution adjustment
        image=args.image
    )
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Get norm layer
    norm = find_norm(model)
    
    # Get number of layers
    num_layers = find_num_hidden_layers(model)
    
    # Get resolutions for this model
    resolutions = get_resolutions(args.model_name)
    
    # Create temporary directory for resized images
    temp_dir = os.path.join(args.output_dir, f"temp_{args.model_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert layer index
    if args.layer_index != 'all':
        target_layer_idx = int(args.layer_index)
    else:
        target_layer_idx = args.layer_index
    
    # Get project root for loading evaluation files
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    # Collect vectors and labels from all resolutions
    all_vectors_by_resolution = {}  # {resolution: [layer][sample][dim]}
    all_labels_by_resolution = {}   # {resolution: [sample_label]}
    resolution_label_map = {}       # {resolution: label_value} for PCA coloring
    
    # Create resolution to label mapping (each resolution gets a unique label)
    for idx, resolution in enumerate(resolutions):
        resolution_str = str(resolution) if resolution else "original"
        resolution_label_map[resolution] = idx
    
    # Process each resolution
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}x{resolution}" if resolution else "Processing original resolution")
        print(f"{'='*60}")
        
        # Extract hidden states for this resolution
        print(f"Extracting hidden states from {len(prompts)} samples...")
        vectors_all_layers = [[] for _ in range(num_layers)]
        resolution_labels = []  # Labels for this resolution (for refusal-based coloring)
        
        for idx, (prompt, img) in enumerate(tqdm(zip(prompts, imgs), total=len(prompts))):
            tmp_img_path = None
            try:
                # Resize image if needed
                if resolution is not None and img is not None:
                    if isinstance(img, str):
                        original_img = Image.open(img).convert("RGB")
                    else:
                        original_img = img.convert("RGB") if hasattr(img, 'convert') else img
                    
                    resized_img = original_img.resize((resolution, resolution), Image.BICUBIC)
                    tmp_img_path = os.path.join(temp_dir, f"tmp_{resolution}_{idx}.png")
                    resized_img.save(tmp_img_path)
                    
                    # Extract hidden states with resized image
                    hidden_states = extract_hidden_states(
                        model, processor, args.model_name, prompt, tmp_img_path, build_prompt
                    )
                else:
                    # Use original image
                    hidden_states = extract_hidden_states(
                        model, processor, args.model_name, prompt, img, build_prompt
                    )
                
                vectors = extract_token_hidden_states(
                    hidden_states, norm, args.token_index, num_layers
                )
                
                for layer_idx in range(num_layers):
                    vectors_all_layers[layer_idx].append(vectors[layer_idx])
                
                # Try to load refusal label for this sample (if Figstep dataset)
                if args.dataset == "Figstep":
                    refusal_labels = load_refusal_labels(args.model_name, args.dataset, resolution, project_root)
                    if refusal_labels is not None and idx < len(refusal_labels):
                        resolution_labels.append(refusal_labels[idx])
                    else:
                        resolution_labels.append(labels[idx] if idx < len(labels) else 0)
                else:
                    resolution_labels.append(labels[idx] if idx < len(labels) else 0)
            finally:
                # Clean up temporary file
                if tmp_img_path and os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
        
        all_vectors_by_resolution[resolution] = vectors_all_layers
        all_labels_by_resolution[resolution] = resolution_labels
    
    # Combine all resolutions for comparison
    print(f"\n{'='*60}")
    print("Combining all resolutions for PCA comparison")
    print(f"{'='*60}")
    
    # Check if we have refusal labels for Figstep
    has_refusal_labels = False
    if args.dataset == "Figstep":
        # Check if refusal labels exist for at least one resolution
        for resolution in resolutions:
            refusal_labels = load_refusal_labels(args.model_name, args.dataset, resolution, project_root)
            if refusal_labels is not None:
                has_refusal_labels = True
                break
    
    # Merge vectors from all resolutions
    combined_vectors_all_layers = [[] for _ in range(num_layers)]
    combined_labels = []  # Combined labels (resolution + refusal if available)
    
    # Create combined label mapping: (resolution_idx * 2 + refusal) for each combination
    combined_label_map = {}  # {(resolution, refusal): label_value}
    label_counter = 0
    
    for resolution in resolutions:
        vectors_all_layers = all_vectors_by_resolution[resolution]
        resolution_str = str(resolution) if resolution else "original"
        
        # Load refusal labels for this resolution if available
        refusal_labels = None
        if has_refusal_labels:
            refusal_labels = load_refusal_labels(args.model_name, args.dataset, resolution, project_root)
        
        for layer_idx in range(num_layers):
            combined_vectors_all_layers[layer_idx].extend(vectors_all_layers[layer_idx])
        
        # Create labels combining resolution and refusal
        if refusal_labels is not None:
            for idx, refusal in enumerate(refusal_labels):
                key = (resolution, refusal)
                if key not in combined_label_map:
                    combined_label_map[key] = label_counter
                    label_counter += 1
                combined_labels.append(combined_label_map[key])
        else:
            # Only resolution-based labeling
            resolution_label = resolution_label_map[resolution]
            combined_labels.extend([resolution_label] * len(vectors_all_layers[0]))
    
    # Generate PCA plot comparing all resolutions
    print(f"Generating PCA plot comparing all resolutions for layer: {target_layer_idx}")
    
    # Create label names
    if has_refusal_labels and combined_label_map:
        # Combined labels: resolution + refusal
        label_names = {}
        for (resolution, refusal), label_val in combined_label_map.items():
            resolution_str = str(resolution) if resolution else "original"
            refusal_str = "Refused" if refusal == 1 else "Not Refused"
            label_names[label_val] = f"Res {resolution_str} - {refusal_str}"
    else:
        # Only resolution-based labels
        label_names = {}
        for resolution, label_val in resolution_label_map.items():
            resolution_str = str(resolution) if resolution else "original"
            label_names[label_val] = f"Res {resolution_str}"
    
    save_path = os.path.join(
        args.output_dir,
        f"{args.model_name}_image_{args.dataset}_pca_{target_layer_idx}_all_resolutions.png"
    )
    
    pca_graph(combined_vectors_all_layers, combined_labels, target_layer_idx, save_path, 
              label_names=label_names)
    
    print(f"✅ PCA plot saved to: {save_path}")
    
    # Optionally, also create refusal-based plots for each resolution if available
    if args.dataset == "Figstep":
        print(f"\n{'='*60}")
        print("Generating refusal-based PCA plots for each resolution")
        print(f"{'='*60}")
        
        for resolution in resolutions:
            refusal_labels = load_refusal_labels(args.model_name, args.dataset, resolution, project_root)
            if refusal_labels is not None:
                vectors_all_layers = all_vectors_by_resolution[resolution]
                resolution_str = str(resolution) if resolution else "original"
                
                save_path = os.path.join(
                    args.output_dir,
                    f"{args.model_name}_image_{args.dataset}_pca_{target_layer_idx}_res_{resolution_str}_refusal.png"
                )
                
                label_names = {0: "Not Refused (False)", 1: "Refused (True)"}
                pca_graph(vectors_all_layers, refusal_labels, target_layer_idx, save_path, label_names=label_names)
                
                print(f"✅ Refusal-based PCA plot saved to: {save_path}")
    
    # Clean up temporary directory
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

