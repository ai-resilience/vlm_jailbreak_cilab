#!/usr/bin/env python3
import argparse
import json

from multimodal_order.config import load_config, resolve_path
from multimodal_order.datasets import load_manifest
from multimodal_order.models.activation_runner import ActivationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract last real input-token activations for Image First and Text First.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--benchmark", required=True); parser.add_argument("--model", required=True)
    args = parser.parse_args(); config, root = load_config(args.config)
    manifest = root / "data" / "manifests" / f"{args.benchmark}_500.json"
    source_root = resolve_path(root, config["paths"]["safe_bench_root" if args.benchmark == "safebench_typo" else "mm_safety_bench_root"])
    rows = load_manifest(manifest, source_root)
    output = resolve_path(root, config["paths"]["output_root"]) / "activations" / args.benchmark / args.model
    alpaca_path = resolve_path(root, config["paths"]["alpaca_prompts"])
    raw = json.loads(alpaca_path.read_text(encoding="utf-8"))
    alpaca = [str(x.get("instruction", x.get("prompt", "")) if isinstance(x, dict) else x) for x in raw]
    ActivationRunner(config["models"][args.model], config["seed"], config["max_new_tokens"]).run_activations(rows, alpaca, output)


if __name__ == "__main__":
    main()
