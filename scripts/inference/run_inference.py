#!/usr/bin/env python3
"""Run inference on a dataset with a VLM model."""
import sys
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.inference import generate_response


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM inference on a dataset")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Figstep, StrongREJECT, XSTest)')
    parser.add_argument('--no_image', action='store_true',
                       help='Use text-only mode')
    parser.add_argument('--image', type=str, default=None,
                       help='Image type to use (blank, panda, noise, etc.)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/inference)')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum tokens to generate')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='System prompt to use (optional)')
    parser.add_argument('--font_size', type=int, default=None,
                       help='Font size for Figstep_font dataset (optional)')
    return parser.parse_args()


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
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    
    # Prepare kwargs for dataset loading
    dataset_kwargs = {}
    if args.font_size is not None:
        dataset_kwargs['font_size'] = args.font_size
    # Pass model_name for Figstep_font dataset to use correct directory
    if args.dataset == 'Figstep_font':
        dataset_kwargs['model_name'] = args.model_name
    
    prompts, labels, imgs, types = load_dataset(
        args.dataset,
        no_image=args.no_image,
        image=args.image,
        **dataset_kwargs
    )
    
    # Determine modality string
    modality = "text_only" if args.no_image else "image"
    
    # Create model/dataset directory structure
    model_dataset_dir = os.path.join(args.output_dir, args.model_name, args.dataset)
    os.makedirs(model_dataset_dir, exist_ok=True)
    
    # Build filename components
    filename_parts = [args.model_name, modality, args.dataset]
    
    # Add font_size to filename if provided
    if args.font_size is not None:
        filename_parts.append(f"font{args.font_size}")
    
    # Include image type in filename if not text-only
    if not args.no_image:
        if args.image == None:
            pass
        else:
            filename_parts.append(args.image)
    
    filename_parts.append("response.jsonl")
    output_file = os.path.join(model_dataset_dir, "_".join(filename_parts))
    
    # Run inference
    print(f"Running inference on {len(prompts)} samples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, img, label in tqdm(zip(prompts, imgs, labels), total=len(prompts)):
            response = generate_response(
                model, processor, args.model_name,
                prompt, img, args.max_new_tokens,
                system_prompt=args.system_prompt
            )
            
            result = {
                'prompt': prompt,
                'response': response,
                'label': label
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"✅ Results saved to: {output_file}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

