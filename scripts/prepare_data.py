#!/usr/bin/env python3
import argparse

from multimodal_order.config import load_config
from multimodal_order.datasets import build_all_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build three stratified 500-example manifests.")
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    config, root = load_config(args.config)
    for path in build_all_manifests(config, root):
        print(path)


if __name__ == "__main__":
    main()
