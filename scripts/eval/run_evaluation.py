#!/usr/bin/env python3
"""Run evaluation on a predictions jsonl with a selected metric."""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.evaluate import (
    load_entries,
    evaluate_with_metric,
    save_evaluation_results,
    write_text_result,
    format_refusal_report,
    format_unsafe_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with a metric")
    parser.add_argument('--metric', type=str, required=True,
                        choices=['keyword', 'llamaguard4', 'beaverdam', 'wildguard'],
                        help='Metric name')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input predictions jsonl/json file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: ../result/evaluation)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set default output directory to external result folder
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'evaluation')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load entries
    print(f"📂 Loading entries from: {args.input}")
    entries = load_entries(args.input)
    print(f"✅ Loaded {len(entries)} entries")

    # Extract dataset name from input filename for special handling
    dataset_name = None
    is_image = False
    image_dataset = None
    
    input_basename = os.path.basename(args.input)
    
    # Check for special datasets
    if "Figstep_res" in input_basename or "figstep_res" in input_basename.lower():
        dataset_name = "Figstep_res"
    elif "Figstep_font" in input_basename or "figstep_font" in input_basename.lower():
        dataset_name = "Figstep_font"
    elif "Figstep" in input_basename or "figstep" in input_basename.lower():
        dataset_name = "Figstep"
    elif "mm_sd_typo" in input_basename:
        dataset_name = "mm_sd_typo"
        image_dataset = "mm_sd_typo"
    elif "mm_typo" in input_basename:
        dataset_name = "mm_typo"
        image_dataset = "mm_typo"
    elif "mm_text" in input_basename:
        dataset_name = "mm_text"
        image_dataset = "mm_text"
    
    # Evaluate
    print(f"🚀 Evaluating with metric: {args.metric}")
    if args.metric in ['llamaguard4', 'beaverdam', 'wildguard'] and dataset_name:
        print(f"📊 Special dataset detected: {dataset_name} (is_image: {is_image})")
        flags, rate = evaluate_with_metric(
            args.metric, 
            entries,
            dataset_name=dataset_name,
            is_image=is_image,
            image_dataset=image_dataset
        )
    else:
        flags, rate = evaluate_with_metric(args.metric, entries)

    # Prepare outputs
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_jsonl = os.path.join(args.output_dir, f"{base_name}_{args.metric}.jsonl")
    out_txt = os.path.join(args.output_dir, f"{base_name}_{args.metric}_result.txt")

    # Determine field/report by metric
    if args.metric == 'keyword':
        result_field = 'refusal'
        report_text = format_refusal_report(total=len(entries), flagged=sum(flags), rate=rate)
    else:  # llamaguard4, beaverdam, wildguard
        result_field = 'unsafe'
        report_text = format_unsafe_report(total=len(entries), flagged=sum(flags), rate=rate)

    # Save results
    print(f"💾 Saving jsonl to: {out_jsonl}")
    save_evaluation_results(entries, flags, out_jsonl, args.metric, result_field=result_field)

    print(f"📝 Saving report to: {out_txt}")
    write_text_result(out_txt, report_text)

    print("✅ Evaluation complete!")


if __name__ == '__main__':
    main()
