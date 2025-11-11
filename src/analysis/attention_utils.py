"""
Utility functions for attention analysis and visualization.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

plt.switch_backend("Agg")


def parse_layers_arg(arg: Optional[str], num_layers: int) -> List[int]:
    if not arg or arg.strip() in {"", "all"}:
        return list(range(num_layers))
    indices: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        layer_idx = int(tok)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        if not (0 <= layer_idx < num_layers):
            raise ValueError(f"Layer index {tok} out of range [0, {num_layers - 1}]")
        indices.append(layer_idx)
    return sorted(set(indices))


def parse_step_indices(arg: Optional[str], num_steps: int) -> List[int]:
    if not arg or arg.strip() == "":
        return list(range(num_steps))
    indices: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        step_idx = int(tok)
        if step_idx <= 0:
            step_idx = num_steps + step_idx - 1
        elif step_idx > 0:
            step_idx = step_idx - 1
            raise ValueError(f"Layer index {step_idx} out of range [0, {num_steps}], input tok {tok}")
        indices.append(step_idx)
    return sorted(set(indices))
    

def head_aggregate(attn: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return attn.mean(dim=0)
    if mode == "max":
        return attn.max(dim=0)[0]
    raise ValueError(f"Unknown head aggregation mode: {mode}")


def tokens_to_strings(tokenizer, ids: List[int]) -> List[str]:
    try:
        return tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        return [str(i) for i in ids]


def topk_indices(x: torch.Tensor, k: int = 20) -> List[int]:
    _, idx = torch.topk(x, k=min(k, x.numel()))
    return idx.tolist()


def guess_image_token_spans(model_name: str, tok_strs: List[str]) -> List[int]:
    spans = set()
    candidate_tokens = [
        "<image>",
        "<image_1>",
        "<image_start>",
        "<img>",
        "</img>",
        "<image_patch>",
        "<IMG_CONTEXT>",
        "<|image_pad|>",
    ]
    tok_array = np.array(tok_strs)
    for tok in candidate_tokens:
        try:
            if tok in tok_strs:
                indices = np.where(tok_array == tok)[0].tolist()
                spans.update(indices)
        except Exception:
            continue
    return sorted(spans)


def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(list(batch), device))
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def generate_with_attn(
    model,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    forward_inputs = move_to_device(inputs, device)

    if "input_ids" not in forward_inputs:
        raise RuntimeError(
            "Attention probing currently expects token-based decoding (inputs include 'input_ids')."
        )

    out = model(
        use_cache=True,
        output_attentions=True,
        return_dict=True,
        **forward_inputs,
    )

    all_attns = torch.stack(out.attentions).squeeze(1).detach()
    input_ids = forward_inputs["input_ids"].detach()
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True).detach()
    return all_attns, input_ids, next_token


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _best_hw_from_K_and_ratio(K: int, target_ratio: float) -> Tuple[int, int]:
    best_h, best_w, best_err = 1, K, float("inf")
    limit = int(math.isqrt(K))
    for h in range(1, limit + 1):
        if K % h != 0:
            continue
        w = K // h
        err = abs((w / h) - target_ratio)
        if err < best_err:
            best_h, best_w, best_err = h, w, err
    return best_h, best_w


def create_attention_overlay(
    attention_matrix_step: np.ndarray,
    img: Image.Image,
    overlay_dir: Path,
    sample_id: str,
    step_idx: int,
    *,
    min_size: int = 448,
    n_cols: int = 6,
    caption_px: int = 10,
    padding: int = 8,
    cmap=plt.cm.viridis,
    alpha_max: float = 0.8,
    alpha_gamma: float = 1.2,
):
    overlay_dir.mkdir(parents=True, exist_ok=True)
    L, K = attention_matrix_step.shape
    Wimg0, Himg0 = img.size

    if min_size > 0 and (Wimg0 % min_size == 0) and (Himg0 % min_size == 0):
        Wbase, Hbase = Wimg0 // min_size, Himg0 // min_size
        if Wbase == 0 or Hbase == 0:
            g = _gcd(Wimg0, Himg0)
            Wbase, Hbase = Wimg0 // g, Himg0 // g
    else:
        g = _gcd(Wimg0, Himg0)
        Wbase, Hbase = Wimg0 // g, Himg0 // g
    target_ratio = (Wbase / Hbase) if Hbase != 0 else (Wimg0 / max(1, Himg0))

    Hgrid, Wgrid = _best_hw_from_K_and_ratio(K, target_ratio)
    patch_h = max(1, int(round(Himg0 / Hgrid)))
    patch_w = max(1, int(round(Wimg0 / Wgrid)))
    tile_H = Hgrid * patch_h
    tile_W = Wgrid * patch_w

    if (tile_W, tile_H) != (Wimg0, Himg0):
        base_rgb = img.convert("RGB").resize((tile_W, tile_H), Image.Resampling.LANCZOS)
        print(
            f"[info] resized image {Wimg0}x{Himg0} → {tile_W}x{tile_H} "
            f"(grid={Hgrid}x{Wgrid}, patch={patch_h}x{patch_w})"
        )
    else:
        base_rgb = img.convert("RGB")

    tiles = []
    for li in range(L):
        att_vec = attention_matrix_step[li]
        att = att_vec.reshape(Hgrid, Wgrid)

        vmin, vmax = float(att.min()), float(att.max())
        att01 = (att - vmin) / (vmax - vmin + 1e-12)

        att_big = np.repeat(np.repeat(att01, patch_h, axis=0), patch_w, axis=1)
        heat_rgb = (cmap(att_big)[..., :3] * 255).astype(np.uint8)
        alpha = (np.clip(att_big, 0, 1) ** alpha_gamma) * alpha_max
        a = alpha[..., None]

        base_np = np.asarray(base_rgb, dtype=np.float32)
        mix = (1 - a) * base_np + a * heat_rgb.astype(np.float32)
        mix = np.clip(mix, 0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(mix))

    cols = max(1, int(n_cols))
    rows = (L + cols - 1) // cols

    cell_w = tile_W + padding
    cell_h = tile_H + caption_px + padding
    panel_w = cols * cell_w + padding
    panel_h = rows * cell_h + padding
    canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(8, caption_px - 2),
        )
    except Exception:
        font = ImageFont.load_default()

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = padding + c * cell_w
        y = padding + r * cell_h
        canvas.paste(tile, (x, y))

        label = f"L{i}"
        tw = draw.textlength(label, font=font)
        th = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]
        cx = x + (tile_W - int(tw)) // 2
        cy = y + tile_H + max(0, (caption_px - th) // 2)
        draw.text((cx, cy), label, fill=(20, 20, 20), font=font)

    save_path = overlay_dir / f"{sample_id}_attn_step{step_idx}.png"
    canvas.save(save_path, dpi=(300, 300))
    print(f"[OK] Saved overlay {save_path}")


def visualize_heatmap(
    attention_matrix: torch.Tensor,
    q_axis: List[str],
    k_axis: List[str],
    step_indices: List[int],
    plot_dir: Path,
    sample_id: str,
    img: Optional[Image.Image] = None,
    overlay_dir: Optional[Path] = None,
    min_size: int = 448,
):
    plot_dir.mkdir(parents=True, exist_ok=True)
    attn_np = attention_matrix.cpu().float().numpy()

    for step_idx in step_indices:
        attention_matrix_step = attn_np[:, step_idx, :]
        y_tok = q_axis[step_idx] if step_idx < len(q_axis) else f"step_{step_idx}"
        plt.figure(figsize=(24, 12))
        plt.imshow(attention_matrix_step, aspect="auto", interpolation="nearest")
        plt.xticks(range(len(k_axis)), k_axis, rotation=90, fontsize=5)
        plt.xlabel("Key Position")
        plt.ylabel("Layer Index")
        plt.title(f"Attention Matrix at step {step_idx} (query: {y_tok})")
        plt.tight_layout()
        save_path = plot_dir / f"{sample_id}_attn_step{step_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        if img is not None and overlay_dir is not None:
            create_attention_overlay(
                attention_matrix_step,
                img,
                overlay_dir,
                sample_id,
                step_idx,
                min_size=min_size,
            )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dirs(base_outdir: Path, args: argparse.Namespace):
    pt_outdir = ensure_dir(base_outdir / "pt") if args.save_pt else None
    pt_img_only_outdir = (
        ensure_dir(base_outdir / "pt_img_only") if args.save_pt and args.image_only else None
    )
    plots_outdir = ensure_dir(base_outdir / "plots") if args.plot else None
    plots_img_only_outdir = ensure_dir(base_outdir / "plots_img_only") if args.plot and args.image_only else None
    overlay_outdir = ensure_dir(base_outdir / "overlay") if args.plot and args.overlay else None
    overlay_img_only_outdir = ensure_dir(base_outdir / "overlay_img_only") if args.plot and args.overlay and args.image_only else None
        
    all_dirs = {
        "pt_outdir": pt_outdir,
        "pt_img_only_outdir": pt_img_only_outdir,
        "plots_outdir": plots_outdir,
        "plots_img_only_outdir": plots_img_only_outdir,
        "overlay_outdir": overlay_outdir,
        "overlay_img_only_outdir": overlay_img_only_outdir,
    }
    

    if args.jsonl_result:
        for dir_name, dir_path_ in all_dirs.items():
            if dir_path_ is not None:
                dir_path = {"true": ensure_dir(dir_path_ / "true"), "false": ensure_dir(dir_path_ / "false")}
                if "plots" in dir_name:
                    true_metadata_path = dir_path_ / "true_metadata.jsonl"
                    false_metadata_path = dir_path_ / "false_metadata.jsonl"
                    dir_path["true_meta"] = open(true_metadata_path, "w", encoding="utf-8")
                    dir_path["false_meta"] = open(false_metadata_path, "w", encoding="utf-8")
                all_dirs[dir_name] = dir_path
                    
                
    return all_dirs


__all__ = [
    "parse_layers_arg",
    "parse_step_indices",
    "head_aggregate",
    "tokens_to_strings",
    "topk_indices",
    "guess_image_token_spans",
    "move_to_device",
    "generate_with_attn",
    "create_attention_overlay",
    "visualize_heatmap",
    "ensure_dir",
    "ensure_dirs",
]

