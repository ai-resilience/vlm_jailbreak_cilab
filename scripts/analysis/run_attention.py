#!/usr/bin/env python3
"""Run attention analysis and visualization for VLM models."""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict
import json
import torch

torch.set_float32_matmul_precision("high")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models import load_model
from src.datasets import load_dataset
from src.inference.processor import build_prompt
from src.analysis.attention_utils import (
    ensure_dirs,
    generate_with_attn,
    guess_image_token_spans,
    head_aggregate,
    move_to_device,
    parse_layers_arg,
    parse_step_indices,
    tokens_to_strings,
    topk_indices,
    visualize_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attention analysis.")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["llava", "llava_next", "intern", "qwen", "deepseek", "deepseek2", "kimi"],
                       help="Model name.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string.")
    parser.add_argument("--image", type=str, default=None,
                       help="Image path or dataset-specific image option.")
    parser.add_argument("--no_image", action="store_true",
                       help="Use text-only mode when loading datasets.")
    parser.add_argument("--step_indices", type=str, default="0",
                       help="Comma-separated step indices (negative for reverse indexing).")
    parser.add_argument("--layers", type=str, default="all",
                       help="Comma-separated layer indices (use 'all' for every layer).")
    parser.add_argument("--heads", type=str, default="mean", choices=["mean", "max"],
                       help="Head aggregation method.")
    parser.add_argument("--image_only", action="store_true",
                       help="Slice attention to image spans if detectable.")
    parser.add_argument("--show", action="store_true",
                       help="Print top tokens per step.")
    parser.add_argument("--plot", action="store_true",
                       help="Save attention heatmaps (and overlays if image available).")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Top-k tokens to display when using --show.")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: ../result/attention).")
    parser.add_argument("--min_size", type=int, default=448,
                       help="Minimum patch size for overlay generation.")
    parser.add_argument("--save_pt", action="store_true",
                        help="Save aggregated attention tensors as .pt files.")
    parser.add_argument("--jsonl_result", type=str, default=None,
                        help="JSONL file containing inference results.")
    parser.add_argument("--normalize_attention", action="store_true",
                        help="Normalize attention to 0-1.")
    parser.add_argument("--overlay", action="store_true",
                        help="Save attention overlays.")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Resolution of input image.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset and not args.prompt:
        raise ValueError("Either --dataset or --prompt must be provided.")

    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        output_root = project_root / "result" / "attention"
    else:
        output_root = Path(args.output_dir)

    base_outdir = output_root / (args.dataset if args.dataset else "custom")
    if args.resolution:
        base_outdir = base_outdir / f"resolution_{args.resolution}"
    
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()

    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        prompts, labels, images, _ = load_dataset(
            args.dataset,
            no_image=args.no_image,
            image=args.image,
        )
    else:
        prompts = [args.prompt]
        labels = [None]
        images = [args.image]

    device = next(model.parameters()).device
    total_samples = len(prompts)
    print(f"Processing {total_samples} samples...")
    
    if args.jsonl_result:
        with open(args.jsonl_result, "r") as f:
            jsonl_entries = [json.loads(line) for line in f]
        if len(jsonl_entries) != total_samples:
            raise ValueError(
                f"jsonl_result entries ({len(jsonl_entries)}) do not match number of prompts ({total_samples})."
            )
    
    all_dirs = ensure_dirs(base_outdir, args)
    pt_outdir = all_dirs["pt_outdir"]
    pt_img_only_outdir = all_dirs["pt_img_only_outdir"]
    plots_outdir_ = all_dirs["plots_outdir"]
    plots_img_only_outdir_ = all_dirs["plots_img_only_outdir"]
    overlay_outdir = all_dirs["overlay_outdir"]
    overlay_img_only_outdir = all_dirs["overlay_img_only_outdir"]
    
    for index, prompt in enumerate(prompts):
        
        if args.jsonl_result:
            sample_json = jsonl_entries[index]
            flagged_bool = sample_json.get("flagged")
            if pt_outdir:
                pt_outdir = pt_outdir["true"] if flagged_bool else pt_outdir["false"]
            if pt_img_only_outdir:
                pt_img_only_outdir = pt_img_only_outdir["true"] if flagged_bool else pt_img_only_outdir["false"]
            if plots_outdir_:
                plots_outdir = plots_outdir_["true"] if flagged_bool else plots_outdir_["false"]
                meta_file = plots_outdir_["true_meta"] if flagged_bool else plots_outdir_["false_meta"]
            if plots_img_only_outdir_:
                plots_img_only_outdir = plots_img_only_outdir_["true"] if flagged_bool else plots_img_only_outdir_["false"]
                meta_file_img_only = plots_img_only_outdir_["true_meta"] if flagged_bool else plots_img_only_outdir_["false_meta"]
            if overlay_outdir:
                overlay_outdir = overlay_outdir["true"] if flagged_bool else overlay_outdir["false"]
            if overlay_img_only_outdir:
                overlay_img_only_outdir = overlay_img_only_outdir["true"] if flagged_bool else overlay_img_only_outdir["false"]

        label = labels[index]
        img_path = images[index] if images else None
        
        print(
            f"\n[{index + 1}/{total_samples}] Prompt: "
            f"{prompt[:60]}{'...' if prompt and len(prompt) > 60 else ''}"
        )

        inputs, attention_mask = build_prompt(
            model,
            processor,
            args.model_name,
            img_path,
            prompt,
        )

        if not hasattr(inputs, "keys"):
            raise RuntimeError(
                "This script expects token-based inputs. The current build_prompt implementation "
                "returned embeddings; please add a token-based path for this model."
            )

        inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
        forward_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            forward_inputs[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        if attention_mask is not None and "attention_mask" not in forward_inputs:
            forward_inputs["attention_mask"] = attention_mask.to(device)

        all_attns, input_ids, next_token = generate_with_attn(model, forward_inputs)
        num_layers, _, num_queries, _ = all_attns.shape

        layer_indices = parse_layers_arg(args.layers, num_layers)
        selected_attn = all_attns[layer_indices]
        # aggregated = torch.stack(
        #     [head_aggregate(layer_attn, args.heads) for layer_attn in selected_attn]
        # )
        aggregated = head_aggregate(selected_attn, args.heads)

        step_indices = parse_step_indices(args.step_indices, num_queries)

        token_ids = input_ids[0].tolist()
        tok_strs = tokens_to_strings(tokenizer, token_ids)
        img_spans = guess_image_token_spans(args.model_name, tok_strs)
        tok_strs_img_only = [tok_strs[i] for i in img_spans]
        aggregated_img_only = aggregated[:, :, img_spans]

        sample_id = f"{index:04d}"
        if label is not None:
            safe_label = str(label).replace(" ", "_")
            sample_id = f"{sample_id}_{safe_label}"

        if args.save_pt:
            if args.image_only:
                torch.save(aggregated_img_only.cpu(), pt_img_only_outdir / f"{sample_id}_attention.pt")
            else:
                torch.save(aggregated.cpu(), pt_outdir / f"{sample_id}_attention.pt")

        # for step_idx in step_indices:
        #     if args.image_only:
        #         step_attn = aggregated_img_only[:, step_idx, :].mean(dim=0)
        #     else:
        #         step_attn = aggregated[:, step_idx, :].mean(dim=0)
        #     top_idx = topk_indices(step_attn, args.top_k)
        #     if args.image_only:
        #         idx_n_tokens = [(i, tok_strs_img_only[i]) if i < len(tok_strs_img_only) else (i, f"<{i}>") for i in top_idx]
        #     else:
        #         idx_n_tokens = [(i, tok_strs[i]) if i < len(tok_strs) else (i, f"<{i}>") for i in top_idx]
        #     print(f"  Step {step_idx}: {' | '.join(str(i) for i, _ in idx_n_tokens)}")

        if args.jsonl_result:
            aggregated_img_only_cpu = aggregated_img_only.cpu()
            # for step_idx in step_indices:
            step_attn = aggregated_img_only_cpu[:, step_indices[0], :] # only the first step temp, TODO: change to all steps
            # top_idx = topk_indices(step_attn, args.top_k)
            img_attn_sum = step_attn.sum(axis=0)
            top_idx = topk_indices(img_attn_sum, args.top_k)
            ret_dict = {"top_idx": top_idx, "img_attn_sum": img_attn_sum[top_idx].tolist()}
            ret_dict = {**ret_dict, **sample_json}
            meta_file_img_only.write(json.dumps(ret_dict) + "\n")
            meta_file_img_only.flush()
        if args.plot:
            pil_img = None
            if img_path and os.path.exists(img_path):
                try:
                    from PIL import Image

                    pil_img = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    warnings.warn(f"Failed to open image {img_path}: {exc}")
            if args.normalize_attention:
                if args.image_only:
                    max_aggregated = aggregated_img_only.max(dim=-1, keepdim=True)[0] # shape: (num_layers, Q)
                    aggregated_img_only = aggregated_img_only / max_aggregated
                else:
                    max_aggregated = aggregated.max(dim=-1, keepdim=True)[0] # shape: (num_layers, Q)
                    aggregated = aggregated / max_aggregated
            
            if args.image_only:
                visualize_heatmap(
                    aggregated_img_only,
                    q_axis=tok_strs,
                    k_axis=tok_strs_img_only,
                    step_indices=step_indices,
                    plot_dir=plots_img_only_outdir,
                    sample_id=sample_id,
                    img=pil_img,
                    overlay_dir=overlay_img_only_outdir,
                    min_size=args.min_size
                )
            # visualize_heatmap(
            #     aggregated,
            #     q_axis=tok_strs,
            #     k_axis=tok_strs,
            #     step_indices=step_indices,
            #     plot_dir=plots_outdir,
            #     sample_id=sample_id,
            #     img=pil_img,
            #     overlay_dir=overlay_outdir,
            #     min_size=args.min_size
            # )

    torch.cuda.empty_cache()
    print("\n✅ Attention analysis completed.")


if __name__ == "__main__":
    main()

