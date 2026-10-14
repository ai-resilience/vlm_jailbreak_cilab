#!/usr/bin/env python3
"""Resize MM-SafetyBench TYPO images: pad to 1024x1024 then resize to 336x336.

Steps:
1) MM_SafetyBench TYPO 이미지를 불러온다.
2) 1024x1024 흰색 배경으로 좌상단 기준 패딩한다.
3) 패딩된 이미지를 336x336으로 리사이즈한다.
4) `dataset/MM_SafetyBench/TYPO_pad1024_to336/{subset}/{subset}_{idx:05d}.png` 로 저장한다.
"""

import argparse
from pathlib import Path

from datasets import load_dataset as load_dataset_
from PIL import Image
from tqdm import tqdm


def pad_then_resize(img: Image.Image, pad_size: int = 1024, target_size: int = 336) -> Image.Image:
    """Pad to (pad_size, pad_size) with white background (top-left anchor), then resize."""
    canvas = Image.new("RGB", (pad_size, pad_size), color="white")
    canvas.paste(img, (0, 0))
    return canvas.resize((target_size, target_size), Image.Resampling.BILINEAR)


def process_subset(subset: str, pad_size: int, target_size: int, save_root: Path) -> None:
    """Process one subset of TYPO split and save images."""
    ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="TYPO")
    images = ds["image"]

    subset_dir = save_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Subset: {subset} | images: {len(images)} | save_dir: {subset_dir}")
    for idx, img in enumerate(tqdm(images, desc=subset)):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        processed = pad_then_resize(img, pad_size=pad_size, target_size=target_size)
        out_path = subset_dir / f"{subset}_{idx:05d}.png"
        processed.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pad MM-SafetyBench TYPO images to 1024x1024 then resize to 336x336."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["HateSpeech", "Illegal_Activitiy"],
        help="Target subsets to resize",
    )
    parser.add_argument(
        "--pad_size",
        type=int,
        default=1024,
        help="Pad size before resizing (square, default: 1024)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=336,
        help="Output resolution after padding (square, default: 336)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory to save resized images "
             "(default: dataset/MM_SafetyBench/TYPO_pad{pad_size}_to{size})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent

    if args.output_dir:
        save_root = Path(args.output_dir)
    else:
        save_root = project_root / "dataset" / "MM_SafetyBench" / f"TYPO_pad{args.pad_size}_to{args.size}"
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Saving padded+resized TYPO images to: {save_root} (pad={args.pad_size}, size={args.size})")
    for subset in args.subsets:
        process_subset(subset, pad_size=args.pad_size, target_size=args.size, save_root=save_root)

    print("✅ Done.")


if __name__ == "__main__":
    main()