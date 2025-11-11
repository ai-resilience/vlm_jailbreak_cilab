#!/usr/bin/env python3
"""Extract and save hidden states with resolution adjustment."""
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.analysis import extract_hidden_states, extract_token_hidden_states
from src.inference.processor import build_prompt
from src.models.base import find_norm, find_num_hidden_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hidden states with resolution adjustment")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use (blank, panda, noise, etc.)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/hidden_states)')
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
        return [280, 560, 840, 1120, 1400, 1680]
    else:  # llava
        return [None]  # Use original resolution


def main():
    args = parse_args()
    
    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'hidden_states')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Token indices to extract
    token_indices = [-1, -2, -3, -4, -5]
    
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
    
    # Create base directory structure: {model_name}/{dataset}/
    base_dir = os.path.join(args.output_dir, args.model_name, args.dataset)
    os.makedirs(base_dir, exist_ok=True)
    
    # Run extraction for each resolution
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"Processing resolution: {resolution}x{resolution}" if resolution else "Processing original resolution")
        print(f"{'='*60}")
        
        # Extract hidden states for this resolution
        print(f"Extracting hidden states from {len(prompts)} samples...")
        
        # Store hidden states for each token index: {token_index: [sample][layer][dim]}
        hidden_states_by_token = {token_idx: [] for token_idx in token_indices}
        
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
                
                # Extract hidden states for each token index
                for token_idx in token_indices:
                    vectors = extract_token_hidden_states(
                        hidden_states, norm, token_idx, num_layers
                    )
                    # Convert to torch tensors and store
                    # vectors is List[np.ndarray], each is [dim]
                    # Stack all layers: [num_layers, dim]
                    vectors_tensor = torch.stack([torch.from_numpy(v) for v in vectors])
                    hidden_states_by_token[token_idx].append(vectors_tensor)
            finally:
                # Clean up temporary file
                if tmp_img_path and os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
        
        # Save hidden states for each token index
        resolution_str = str(resolution) if resolution else "original"
        
        for token_idx in token_indices:
            # Create token index directory: {model_name}/{dataset}/{token_index}/
            token_dir = os.path.join(base_dir, str(token_idx))
            os.makedirs(token_dir, exist_ok=True)
            
            # Stack all samples: [num_samples, num_layers, dim]
            all_hidden_states = torch.stack(hidden_states_by_token[token_idx])
            
            # Save to .pt file
            save_path = os.path.join(
                token_dir,
                f"{args.model_name}_{args.dataset}_res_{resolution_str}.pt"
            )
            
            torch.save(all_hidden_states, save_path)
            print(f"✅ Saved hidden states for token {token_idx} to: {save_path}")
            print(f"   Shape: {all_hidden_states.shape} (samples, layers, dim)")
    
    # Clean up temporary directory
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    torch.cuda.empty_cache()
    print(f"\n✅ All hidden states extracted and saved!")


if __name__ == '__main__':
    main()

