#!/usr/bin/env python3
"""Lightweight pilot: core tests plus synthetic evaluation and all three figures."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from multimodal_order.evaluation.pipeline import evaluate_file
from multimodal_order.visualization import figure2_wildguard_asr, figure3_layer_cosine, figure4_pca


def main() -> None:
    root = Path(__file__).resolve().parents[1]; work = root / "results" / ".pilot"
    if work.exists(): shutil.rmtree(work)
    try:
        response = work / "responses" / "safebench_typo" / "pilot" / "image_first.json"
        response.parent.mkdir(parents=True); response.write_text(json.dumps({"metadata": {"benchmark": "safebench_typo", "model": "pilot"}, "rows": [{"category_id": 0, "task_id": i, "question": "q", "response": "<think>omit</think>\n1. answer"} for i in range(8)]}), encoding="utf-8")
        evaluation = work / "evaluations" / "safebench_typo" / "pilot" / "image_first.json"
        evaluate_file(response, evaluation, {}, run_model_judges=False)
        text_eval = evaluation.with_name("text_first.json"); text_eval.write_text(evaluation.read_text(encoding="utf-8"), encoding="utf-8")
        data = json.loads(evaluation.read_text(encoding="utf-8")); data["summary"]["wildguard_asr"] = .5; evaluation.write_text(json.dumps(data), encoding="utf-8"); text_eval.write_text(json.dumps(data), encoding="utf-8")
        acts = work / "activations" / "safebench_typo" / "pilot"; acts.mkdir(parents=True)
        rng = np.random.default_rng(42)
        for name, count in [("alpaca", 10), ("text_only", 10), ("image_first", 10), ("text_first", 10)]: np.save(acts / f"{name}.npy", rng.normal(size=(count, 4, 8)).astype("float32"))
        figures = work / "figures"
        figure2_wildguard_asr(work / "evaluations", figures / "figure2_wildguard_asr")
        figure3_layer_cosine(work / "activations", figures / "figure3_layer_cosine")
        figure4_pca(work / "activations", figures / "figure4_pca")
        assert all((figures / name).with_suffix(".png").is_file() for name in ("figure2_wildguard_asr", "figure3_layer_cosine", "figure4_pca"))
        print("Pilot test passed")
    finally:
        if work.exists(): shutil.rmtree(work)


if __name__ == "__main__":
    main()
