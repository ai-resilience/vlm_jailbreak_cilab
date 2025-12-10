#!/usr/bin/env python3
"""Token intervention: Mask attention from user_input_end_pos tokens to image tokens."""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.analysis.token_intervention import generate_with_token_intervention


def parse_args():
    parser = argparse.ArgumentParser(
        description="Token intervention: Mask attention from user_input_end_pos to image tokens"
    )
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['llava', 'llava_next', 'intern', 'qwen', 'deepseek', 'deepseek2', 'kimi'],
                       help='Model name')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image (optional)')
    parser.add_argument('--mask_image_tokens', action='store_true',
                       help='Automatically mask all image tokens')
    parser.add_argument('--mask_only_decoding', action='store_true',
                       help='Only mask during decoding stage (not prefilling)')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/token_intervention)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'token_intervention')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model in eager mode to get attention weights (not flash attention)
    print(f"Loading model: {args.model_name} (eager mode for attention weights)")
    model, processor, tokenizer = load_model(args.model_name, attn_implementation="eager")
    model.eval()
    
    # Generate with intervention
    print(f"\n{'='*60}")
    print(f"Token Intervention: Mask attention to image tokens")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print(f"Image: {args.image}")
    print(f"Mask image tokens: {args.mask_image_tokens}")
    print(f"Mask only decoding: {args.mask_only_decoding}")
    print(f"{'='*60}\n")
    
    result = generate_with_token_intervention(
        model=model,
        processor=processor,
        model_name=args.model_name,
        prompt=args.prompt,
        img=args.image,
        mask_image_tokens=args.mask_image_tokens,
        mask_only_decoding=args.mask_only_decoding,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"✅ Generation complete!")
    print(f"{'='*60}")
    print(f"Response: {result['response']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
