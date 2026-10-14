#!/usr/bin/env python3
import argparse

from multimodal_order.config import load_config, resolve_path
from multimodal_order.constants import CONDITIONS
from multimodal_order.evaluation.pipeline import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Refusal, Target String, LlamaGuard, WildGuard, and EM.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rules-only", action="store_true", help="Pilot mode without learned judges")
    args = parser.parse_args()
    config, root = load_config(args.config)
    result_root = resolve_path(root, config["paths"]["output_root"])
    for condition in CONDITIONS:
        source = result_root / "responses" / args.benchmark / args.model / f"{condition.value}.json"
        output = result_root / "evaluations" / args.benchmark / args.model / f"{condition.value}.json"
        result = evaluate_file(source, output, config["judges"], not args.rules_only, config.get("batch_size", 1))
        print(condition.value, result["summary"])


if __name__ == "__main__":
    main()
