#!/usr/bin/env python3
"""Run inference on MM-SafetyBench TYPO images with white padding.

- 입력: 리사이즈된 TYPO 이미지(예: 336x336) 디렉터리
- 규칙: 모델별 타깃 해상도보다 작은 경우 좌상단에 원본을 붙이고, 나머지는 흰색으로 패딩
- 출력: resolution별 JSONL (`{model}_image_mm_typo_response_{resolution}.jsonl`)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import tempfile

import torch
from datasets import load_dataset as load_dataset_
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.inference import generate_response
from src.models import load_model


def get_resolutions(model_name: str) -> list:
    """Get resolution list for each model (preserve aspect ratio)."""
    if model_name in ["intern", "phi"]:
        return [448, 896, 1344]
    if model_name == "llava_next":
        return [336, 672]
    if model_name in ["deepseek2", "deepseek"]:
        return [384, 768, 1152]
    if model_name in ["qwen", "kimi"]:
        return [280, 560, 840, 1120, 1400, 1680]
    return [672]  # Default fallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLM inference on MM-SafetyBench TYPO images with padding"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--resized_dir",
        type=str,
        default=None,
        help="Path to resized TYPO images (default: dataset/MM_SafetyBench/TYPO_pad1024_to336)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ../result/inference/mm_typo)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["HateSpeech", "Illegal_Activitiy"],
        help="Subsets to process (TYPO split)",
    )
    return parser.parse_args()


def pad_to_square_top_left(img: Image.Image, target: int) -> Image.Image:
    """Pad image to (target, target) with white background, anchored top-left."""
    canvas = Image.new("RGB", (target, target), color="white")
    canvas.paste(img, (0, 0))
    return canvas


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Set default paths
    resized_dir = Path(args.resized_dir) if args.resized_dir else project_root / "dataset" / "MM_SafetyBench" / "TYPO_pad1024_to336"
    output_root = Path(args.output_dir) if args.output_dir else project_root.parent / "result" / "inference" / "mm_typo"

    if not resized_dir.exists():
        raise ValueError(f"Resized TYPO directory not found: {resized_dir}")

    model_out_dir = output_root / args.model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()

    resolutions = get_resolutions(args.model_name)

    for resolution in resolutions:
        out_jsonl = model_out_dir / f"{args.model_name}_image_mm_typo_response_padded_{resolution}.jsonl"
        print(f"\n📁 Resolution {resolution} -> {out_jsonl}")

        with open(out_jsonl, "w", encoding="utf-8") as f_out:
            for subset in args.subsets:
                subset_dir = resized_dir / subset
                if not subset_dir.exists():
                    print(f"  ⚠️  Subset not found: {subset_dir}, skip")
                    continue

                # Load text prompts for alignment (TYPO split to match images)
                ds_text = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="TYPO")

                image_files = sorted(
                    str(subset_dir / fname)
                    for fname in os.listdir(subset_dir)
                    if fname.lower().endswith(".png")
                )
                if not image_files:
                    print(f"  ⚠️  No images in {subset_dir}, skip")
                    continue

                length = min(len(image_files), len(ds_text["question"]))
                for idx in tqdm(range(length), desc=f"{subset}-size{resolution}", leave=False):
                    img_path = image_files[idx]
                    prompt = ds_text["question"][idx]

                    try:
                        img = Image.open(img_path).convert("RGB")
                        padded = pad_to_square_top_left(img, resolution)

                        # Save to temporary file (auto-cleaned), no persistent padded storage
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            padded.save(tmp.name)
                            padded_path = tmp.name

                        response = generate_response(
                            model,
                            processor,
                            args.model_name,
                            prompt,
                            padded_path,
                            args.max_new_tokens,
                        )

                        # Clean up temp file
                        try:
                            os.remove(padded_path)
                        except OSError:
                            pass

                        result = {
                            "image_path": img_path,  # original resized image path
                            "prompt": prompt,
                            "response": response,
                            "label": 0,  # All are unsafe
                            "resolution": resolution,
                            "subset": subset,
                            "orig_image": img_path,
                            "padded_temp": True,
                        }
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                    except Exception as e:
                        print(f"❌ Error processing {img_path}: {e}")
                        continue

        print(f"✅ Saved: {out_jsonl}")

    torch.cuda.empty_cache()
    print(f"\n✅ All inference completed for model: {args.model_name}")


if __name__ == "__main__":
    main()

