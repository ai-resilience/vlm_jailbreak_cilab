#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tidy ASR table from evaluation JSON files.")
    parser.add_argument("--results-dir", default="results/evaluations")
    parser.add_argument("--output", default="results/tables/asr.csv")
    args = parser.parse_args()
    rows = []
    for path in Path(args.results_dir).glob("*/*/*.json"):
        data = json.loads(path.read_text(encoding="utf-8")); meta = data.get("metadata", {})
        label = {"image_first": "Image First", "text_first": "Text First"}.get(path.stem, path.stem)
        rows.append({"benchmark": meta.get("benchmark", path.parents[1].name), "model": meta.get("model", path.parent.name), "condition": label, "n": data["total"], **data["summary"]})
    if not rows:
        raise SystemExit("No evaluation JSON files found")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: (x["benchmark"], x["model"], x["condition"])))
    print(output)


if __name__ == "__main__":
    main()
