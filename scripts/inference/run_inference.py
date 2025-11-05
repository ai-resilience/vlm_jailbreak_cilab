#!/usr/bin/env python3
"""Run inference on a dataset with a VLM model."""
import sys
import os
import argparse
import json
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.datasets import load_dataset
from src.inference import generate_response


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM inference on a dataset")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Figstep, StrongREJECT, XSTest)')
    parser.add_argument('--no_image', action='store_true',
                       help='Use text-only mode')
    parser.add_argument('--image', type=str, default='blank',
                       help='Image type to use (blank, panda, noise, etc.)')
    parser.add_argument('--output_dir', type=str, default='./result/inference',
                       help='Output directory')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum tokens to generate')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    prompts, labels, imgs, types = load_dataset(
        args.dataset,
        no_image=args.no_image,
        image=args.image
    )
    
    # Determine modality string
    modality = "text_only" if args.no_image else "image"
    
    # Output file
    output_file = os.path.join(
        args.output_dir,
        f"{args.model_name}_{modality}_{args.dataset}_response.jsonl"
    )
    
    # Run inference
    print(f"Running inference on {len(prompts)} samples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, img, label in tqdm(zip(prompts, imgs, labels), total=len(prompts)):
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
    
    print(f"✅ Results saved to: {output_file}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

