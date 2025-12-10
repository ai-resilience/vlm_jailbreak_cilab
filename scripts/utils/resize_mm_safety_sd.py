#!/usr/bin/env python3
"""Resize MM-SafetyBench SD split images to 224x224 and save locally.

기능:
- PKU-Alignment/MM-SafetyBench 의 SD split 이미지를 불러와 224x224로 리사이즈
- `dataset/MM_SafetyBench/SD_resized/{subset}/{subset}_{idx:05d}.png` 형태로 저장

사용 예시:
    python scripts/utils/resize_mm_safety_sd.py --subsets HateSpeech Illegal_Activitiy --size 224
"""

import argparse
from pathlib import Path

from datasets import load_dataset as load_dataset_
from PIL import Image
from tqdm import tqdm


def resize_and_save_subset(subset: str, size: int, save_root: Path) -> None:
    """Resize images for one subset and save them."""
    ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="TYPO")
    images = ds["image"]

    subset_dir = save_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Subset: {subset} | images: {len(images)} | save_dir: {subset_dir}")
    for idx, img in enumerate(tqdm(images, desc=subset)):
        if not isinstance(img, Image.Image):
            # Defensive: convert to PIL Image if needed
            img = Image.fromarray(img)
        resized = img.resize((size, size), Image.Resampling.BILINEAR)
        out_path = subset_dir / f"{subset}_{idx:05d}.png"
        resized.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize MM-SafetyBench SD images and save locally."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["HateSpeech", "Illegal_Activitiy"],
        help="Target subsets to resize",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output resolution (square)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory to save resized images (default: dataset/MM_SafetyBench/SD_resized)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    default_out = project_root / "dataset" / "MM_SafetyBench" / "SD_resized"
    save_root = Path(args.output_dir) if args.output_dir else default_out
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Saving resized images to: {save_root} (size={args.size})")
    for subset in args.subsets:
        resize_and_save_subset(subset, args.size, save_root)

    print("✅ Done.")


if __name__ == "__main__":
    main()