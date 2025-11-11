#!/usr/bin/env python3
"""Run inference on a dataset with a VLM model, with resolution adjustment."""
import sys
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.inference import generate_response


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM inference on a dataset with resolution adjustment")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Figstep, StrongREJECT, XSTest)')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use (blank, panda, noise, etc.)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/inference)')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum tokens to generate')
    return parser.parse_args()


def get_resolutions(model_name: str) -> list:
    """Get resolution list for each model.(preserve aspect ratio)"""
    if model_name == "intern":
        return [448, 896, 1344]
    elif model_name == "llava_next":
        return [336, 672]
    elif model_name == "deepseek2" or model_name == "deepseek":
        return [384, 768, 1152]
    elif model_name == "qwen" or model_name == "kimi":
        return [280, 560, 840, 1120, 1400, 1680]
    else:  # llava, deepseek
        return [None]  # Use original resolution


def main():
    args = parse_args()
    
    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'inference')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Determine if dataset has its own images
    # mm_safety datasets (mm_text, mm_typo, mm_sd_typo) and Figstep have their own images
    datasets_with_images = ["mm_text", "mm_typo", "mm_sd_typo", "Figstep"]
    has_dataset_images = args.dataset in datasets_with_images
    
    # Always use image mode (resolution adjustment requires images)
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    prompts, labels, imgs, types = load_dataset(
        args.dataset,
        no_image=False,  # Always use images for resolution adjustment
        image=args.image
    )
    
    # Always image mode
    modality = "image"
    
    # Get resolutions for this model
    resolutions = get_resolutions(args.model_name)
    
    # Create temporary directory for resized images
    temp_dir = os.path.join(args.output_dir, f"temp_{args.model_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create model/dataset directory structure
    model_dataset_dir = os.path.join(args.output_dir, args.model_name, args.dataset)
    os.makedirs(model_dataset_dir, exist_ok=True)
    
    # Run inference for each resolution
    for resolution in resolutions:
        # Image mode: process each resolution
        output_file = os.path.join(
            model_dataset_dir,
            f"{args.model_name}_{modality}_{args.dataset}_response_{resolution}.jsonl"
        )
        
        print(f"Running inference with resolution {resolution}x{resolution}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, (prompt, img, label) in enumerate(tqdm(zip(prompts, imgs, labels), total=len(prompts))):
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
                        
                        # Run inference with resized image
                        response = generate_response(
                            model, processor, args.model_name,
                            prompt, tmp_img_path, args.max_new_tokens
                        )
                    else:
                        # Use original image
                        response = generate_response(
                            model, processor, args.model_name,
                            prompt, img, args.max_new_tokens
                        )
                    
                    result = {
                        'prompt': prompt,
                        'response': response,
                        'label': label
                    }
                    
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                finally:
                    # Clean up temporary file
                    if tmp_img_path and os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)
        
        print(f"✅ Results saved to: {output_file}")
    
    # Clean up temporary directory
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
