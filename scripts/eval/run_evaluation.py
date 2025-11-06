#!/usr/bin/env python3
"""Run evaluation on a predictions jsonl with a selected metric."""
import sys
import os
import argparse

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
                        choices=['keyword', 'llamaguard4'],
                        help='Metric name')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input predictions jsonl/json file')
    parser.add_argument('--output_dir', type=str, default='./result/eval',
                        help='Output directory for evaluation results')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load entries
    print(f"📂 Loading entries from: {args.input}")
    entries = load_entries(args.input)
    print(f"✅ Loaded {len(entries)} entries")

    # Evaluate
    print(f"🚀 Evaluating with metric: {args.metric}")
    flags, rate = evaluate_with_metric(args.metric, entries)

    # Prepare outputs
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_jsonl = os.path.join(args.output_dir, f"{base_name}_{args.metric}.jsonl")
    out_txt = os.path.join(args.output_dir, f"{base_name}_{args.metric}_result.txt")

    # Determine field/report by metric
    if args.metric == 'keyword':
        result_field = 'refusal'
        report_text = format_refusal_report(total=len(entries), flagged=sum(flags), rate=rate)
    else:  # llamaguard4
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
