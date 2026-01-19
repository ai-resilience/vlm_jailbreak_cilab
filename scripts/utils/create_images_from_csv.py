#!/usr/bin/env python3
"""
Create tiled images from SafeBench-Tiny.csv instructions.
Splits each instruction into 4 parts and creates a 2x2 tiled image.

python scripts/utils/create_images_from_csv.py \
    --csv dataset/FigStep/data/question/safebench.csv \
    --model_name llava_next \
    --tile_size 336 \
    --image_size 672 \
    --stego_indices 0 1 2 3 \
    --amplitude 120 \
    --font_size 40
"""
import argparse
import csv
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.utils.create_tiled_text_image_partial_stego import create_tiled_text_image


def split_instruction_into_4_parts(instruction: str) -> list:
    """
    Split instruction text into 4 parts for 2x2 tiled image.
    
    Strategy:
    1. All words distributed evenly across 4 tiles
    
    Args:
        instruction: Instruction text to split
        
    Returns:
        List of 4 text strings
    """
    # Remove leading/trailing whitespace and periods
    instruction = instruction.strip().rstrip('.')
    
    # Split into words
    words = instruction.split()
    
    if len(words) == 0:
        return ["", "", "", ""]
    
    # Distribute all words evenly across 4 tiles
    total_words = len(words)
    words_per_tile = total_words // 4
    remainder = total_words % 4
    
    parts = []
    start_idx = 0
    
    for i in range(4):
        # Distribute remainder words to first tiles
        part_size = words_per_tile + (1 if i < remainder else 0)
        end_idx = start_idx + part_size
        
        part_words = words[start_idx:end_idx]
        # Join words with space (horizontal layout)
        part_text = " ".join(part_words) if part_words else ""
        
        # Add part text to parts list
        parts.append(part_text)
        
        start_idx = end_idx
    
    return parts


def process_csv_to_images(
    csv_path: str,
    output_dir: str,
    instruction_column: str = "instruction",
    exclude_categories: list = None,
    **image_kwargs
):
    """
    Process CSV file and create tiled images for each instruction.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save output images
        instruction_column: Name of the column containing instructions
        exclude_categories: List of category names to exclude (default: ["Legal Opinion", "Financial Advice", "Health Consultation"])
        **image_kwargs: Additional arguments for create_tiled_text_image
    """
    # Default excluded categories
    if exclude_categories is None:
        exclude_categories = ["Legal Opinion", "Financial Advice", "Health Consultation"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"📖 Loaded {len(rows)} rows from {csv_path}")
    
    # Filter out excluded categories
    filtered_rows = []
    for row in rows:
        category_name = row.get('category_name', '').strip()
        if category_name not in exclude_categories:
            filtered_rows.append(row)
    
    print(f"🔍 Filtered to {len(filtered_rows)} rows (excluded categories: {exclude_categories})")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Process each row
    for idx, row in enumerate(filtered_rows, start=1):
        # Get instruction
        instruction = row.get(instruction_column, "").strip()
        
        if not instruction:
            print(f"⚠️  Row {idx}: No instruction found, skipping")
            continue
        
        # Split instruction into 4 parts
        parts = split_instruction_into_4_parts(instruction)
        
        # Create output filename
        # Use dataset, category_id, task_id if available
        dataset = row.get('dataset', 'unknown')
        category_id = row.get('category_id', '0')
        task_id = row.get('task_id', '0')
        output_filename = f"{dataset}_cat{category_id}_task{task_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create tiled image
        print(f"[{idx}/{len(rows)}] Creating image: {output_filename}")
        print(f"  Instruction: {instruction}")
        print(f"  Parts: {parts}")
        
        try:
            create_tiled_text_image(
                output_path=output_path,
                texts=parts,
                **image_kwargs
            )
            print(f"  ✅ Saved: {output_path}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    print(f"✅ Completed! Created {len(filtered_rows)} images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create tiled images from CSV instructions"
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., phi, deepseek2) - images will be saved to dataset/FigStep/data/images/SafeBench/{model_name}'
    )
    parser.add_argument(
        '--instruction_column',
        type=str,
        default='instruction',
        help='Name of instruction column (default: instruction)'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        default=None,
        help='Size of each tile (TODO: set per model)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help='Total image size (TODO: set per model)'
    )
    parser.add_argument(
        '--background',
        type=str,
        default="white",
        help='Background color (default: white)'
    )
    parser.add_argument(
        '--text_color',
        type=str,
        default="blue",
        help='Text color for normal tiles (default: black)'
    )
    parser.add_argument(
        '--font_size',
        type=int,
        default=30,
        help='Font size (default: 30)'
    )
    parser.add_argument(
        '--font_path',
        type=str,
        default=None,
        help='Path to font file (optional)'
    )
    parser.add_argument(
        '--amplitude',
        type=int,
        default=50,
        help='Pattern intensity for steganography (default: 15)'
    )
    parser.add_argument(
        '--line_spacing',
        type=float,
        default=1.2,
        help='Line spacing multiplier (default: 1.2)'
    )
    parser.add_argument(
        '--stego_indices',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3],
        help='Tile indices (0-indexed) to use steganography (default: 0 1 2 3 for all tiles)'
    )
    
    args = parser.parse_args()
    
    # Set tile_size and image_size based on model_name if not provided
    # TODO: Add appropriate values for each model
    if args.tile_size is None or args.image_size is None:
        model_configs = {
            # TODO: Add configurations for each model
            # Example format:
            'intern': {'tile_size': 448, 'image_size': 896},
            'deepseek2': {'tile_size': 384, 'image_size': 768},
            'llava_next': {'tile_size': 336, 'image_size': 672},
        }
        
        if args.model_name in model_configs:
            config = model_configs[args.model_name]
            args.tile_size = config.get('tile_size')
            args.image_size = config.get('image_size')
            print(f"✅ Using default config for {args.model_name}: tile_size={args.tile_size}, image_size={args.image_size}")
        else:
            print("❌ Error: --tile_size and --image_size are required")
            print(f"   TODO: Add configuration for model '{args.model_name}' in model_configs dictionary")
            print("   Example:")
            print(f"     '{args.model_name}': {{'tile_size': <size>, 'image_size': <size>}},")
            sys.exit(1)
    
    # Build output directory path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "dataset" / "FigStep" / "data" / "images" / "SafeBench" / args.model_name
    
    # Prepare image kwargs
    image_kwargs = {
        'tile_size': args.tile_size,
        'image_size': args.image_size,
        'background_color': args.background,
        'text_color': args.text_color,
        'font_size': args.font_size,
        'font_path': args.font_path,
        'amplitude': args.amplitude,
        'line_spacing': args.line_spacing,
        'stego_tile_indices': args.stego_indices
    }
    
    # Process CSV
    process_csv_to_images(
        csv_path=args.csv,
        output_dir=str(output_dir),
        instruction_column=args.instruction_column,
        **image_kwargs
    )


if __name__ == '__main__':
    main()
