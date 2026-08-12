import json
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure2_wildguard_asr(evaluation_root: Path, output: Path) -> None:
    records = []
    for path in evaluation_root.glob("*/*/*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        if "wildguard_asr" in data.get("summary", {}):
            records.append((meta.get("model", path.parent.name), meta.get("benchmark", path.parent.parent.name), path.stem, 100 * data["summary"]["wildguard_asr"]))
    if not records:
        raise FileNotFoundError(f"no evaluation summaries under {evaluation_root}")
    models = sorted({x[0] for x in records}); benchmarks = sorted({x[1] for x in records})
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 5), squeeze=False)
    for ax, benchmark in zip(axes[0], benchmarks):
        x = np.arange(len(models)); width = 0.36
        for offset, condition, label in [(-width/2, "image_first", "Image First"), (width/2, "text_first", "Text First")]:
            values = [next((r[3] for r in records if r[:3] == (m, benchmark, condition)), np.nan) for m in models]
            ax.bar(x + offset, values, width, label=label)
        ax.set_title(benchmark.replace("_", " ")); ax.set_xticks(x, models, rotation=25, ha="right"); ax.set_ylim(0, 100); ax.set_ylabel("WildGuard ASR (%)")
    axes[0, 0].legend(); _save(fig, output)


def _cosine_by_layer(vector: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sum(vector * target, axis=1) / (np.linalg.norm(vector, axis=1) * np.linalg.norm(target, axis=1) + 1e-12)


def figure3_layer_cosine(activation_root: Path, output: Path) -> None:
    dirs = sorted(p for p in activation_root.glob("*/*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no activation directories under {activation_root}")
    benchmarks = sorted({p.parent.name for p in dirs}); models = sorted({p.name for p in dirs})
    fig, axes = plt.subplots(len(benchmarks), len(models), figsize=(5 * len(models), 4 * len(benchmarks)), squeeze=False)
    for directory in dirs:
        ax = axes[benchmarks.index(directory.parent.name), models.index(directory.name)]
        harmless = np.load(directory / "alpaca.npy").mean(axis=0)
        baseline = np.load(directory / "text_only.npy").mean(axis=0)
        for filename, label in [("image_first.npy", "Image First"), ("text_first.npy", "Text First")]:
            direction = np.load(directory / filename).mean(axis=0) - baseline
            ax.plot(np.arange(1, direction.shape[0] + 1), _cosine_by_layer(direction, -harmless), label=label)
        ax.set_title(f"{directory.parent.name}\n{directory.name}"); ax.set_xlabel("Layer"); ax.set_ylabel("Cosine similarity to −refusal"); ax.axhline(0, color="grey", lw=.8)
    axes[0, 0].legend(); _save(fig, output)


def figure4_pca(activation_root: Path, output: Path, layer: int = -1) -> None:
    dirs = sorted(p for p in activation_root.glob("*/*") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no activation directories under {activation_root}")
    benchmarks = sorted({p.parent.name for p in dirs}); models = sorted({p.name for p in dirs})
    fig, axes = plt.subplots(len(benchmarks), len(models), figsize=(5 * len(models), 4 * len(benchmarks)), squeeze=False)
    for directory in dirs:
        ax = axes[benchmarks.index(directory.parent.name), models.index(directory.name)]
        arrays = {name: np.load(directory / f"{name}.npy")[:, layer, :] for name in ("alpaca", "text_only", "image_first", "text_first")}
        fit = np.concatenate([arrays["alpaca"], arrays["text_only"]], axis=0)
        pca = PCA(n_components=2, random_state=0).fit(fit)
        for name, color in [("alpaca", "green"), ("text_only", "red"), ("image_first", "royalblue"), ("text_first", "orange")]:
            points = pca.transform(arrays[name]); ax.scatter(points[:, 0], points[:, 1], s=8, alpha=.5, label=name.replace("_", " "))
        ax.set_title(f"{directory.parent.name}\n{directory.name}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    axes[0, 0].legend(); _save(fig, output)
