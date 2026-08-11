#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifests and response artifacts.")
    parser.add_argument("--root", default="."); parser.add_argument("--expected", type=int, default=500)
    args = parser.parse_args(); root = Path(args.root).resolve(); errors = []
    for manifest in (root / "data" / "manifests").glob("*.json"):
        rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
        ids = [(x.get("category_id"), x.get("task_id")) for x in rows]
        if len(rows) != args.expected: errors.append(f"{manifest}: expected {args.expected}, got {len(rows)}")
        if len(ids) != len(set(ids)): errors.append(f"{manifest}: duplicate IDs")
        if any(Path(x["image_path"]).is_absolute() for x in rows): errors.append(f"{manifest}: absolute image path")
    for model_dir in (root / "results" / "responses").glob("*/*"):
        for name in ("image_first", "text_first"):
            path = model_dir / f"{name}.json"
            if not path.is_file(): errors.append(f"missing {path}"); continue
            data = json.loads(path.read_text(encoding="utf-8")); rows = data.get("rows", data)
            if len(rows) != args.expected: errors.append(f"{path}: expected {args.expected}, got {len(rows)}")
            if any(not str(x.get("response", "")).strip() for x in rows): errors.append(f"{path}: empty response")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Artifact validation passed")


if __name__ == "__main__":
    main()
