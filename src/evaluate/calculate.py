import argparse
import os
from .base import (
    load_entries,
    refusal_rate_from_field,
    format_refusal_report,
    format_unsafe_report,
    write_text_result,
)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predict_jsonl',
        type=str,
        required=True,
        help='Path to the input JSON file.',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='refusal',
        choices=['refusal', 'unsafe'],
        help='Metric to aggregate: "refusal" (Refusal Rate) or "unsafe" (Unsafe Rate).',
    )
    return parser.parse_args()

def compute_asr(jsonl_path: str, metric: str = 'unsafe') -> None:
    entries = load_entries(jsonl_path)
    total = len(entries)
    rate = refusal_rate_from_field(entries, field=metric)
    flagged = int(round(rate * total))

    # Save
    base, _ = os.path.splitext(jsonl_path)
    out_txt = f"{base}_result.txt"
    if metric == 'unsafe':
        report = format_unsafe_report(total=total, flagged=flagged, rate=rate)
    else:
        report = format_refusal_report(total=total, flagged=flagged, rate=rate)
    write_text_result(out_txt, report)

def main() -> None:
    args = parse_arguments()
    compute_asr(args.predict_jsonl, metric=args.metric)

if __name__ == '__main__':
    main()

