#!/usr/bin/env python3
import argparse
from pathlib import Path

from multimodal_order.config import load_config, resolve_path
from multimodal_order.visualization import figure2_wildguard_asr, figure3_layer_cosine, figure4_pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper Figures 2, 3, and 4.")
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    config, root = load_config(args.config); results = resolve_path(root, config["paths"]["output_root"])
    out = results / "figures"
    figure2_wildguard_asr(results / "evaluations", out / "figure2_wildguard_asr")
    figure3_layer_cosine(results / "activations", out / "figure3_layer_cosine")
    figure4_pca(results / "activations", out / "figure4_pca")


if __name__ == "__main__":
    main()
