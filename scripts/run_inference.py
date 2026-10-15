#!/usr/bin/env python3
import argparse
from pathlib import Path

from multimodal_order.config import load_config, resolve_path
from multimodal_order.datasets import load_manifest
from multimodal_order.models import InferenceRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Image First and Text First inference.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.benchmark not in config["benchmarks"] or args.model not in config["models"]:
        parser.error("unknown benchmark or model key")
    manifest = root / "data" / "manifests" / f"{args.benchmark}_500.json"
    source_root = resolve_path(root, config["paths"]["safe_bench_root" if args.benchmark == "safebench_typo" else "mm_safety_bench_root"])
    rows = load_manifest(manifest, source_root)
    output = resolve_path(root, config["paths"]["output_root"]) / "responses" / args.benchmark / args.model
    runner = InferenceRunner(config["models"][args.model], config["seed"], config["max_new_tokens"])
    runner.run(rows, output, args.model, args.benchmark)


if __name__ == "__main__":
    main()
