"""MultiTrust dataset loader (Hugging Face: thu-ml/MultiTrust)."""
from typing import Tuple, List, Optional, Any
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class MultiTrustDataset(BaseDataset):
    """Loader for the `thu-ml/MultiTrust` dataset from Hugging Face.

    This class aims to mirror the style of the project's other dataset loaders
    (see `advbench.py` and `jailbreakv28k.py`). It attempts to be flexible with
    the column names used by the upstream dataset: it will try several
    candidates for text/prompt and for label columns.

    Args:
        no_image: If True, do not attempt to use dataset image fields.
        image: If provided, override image paths with this value.
        split: Dataset split to load (commonly 'train' or 'test').
        use_hf: Whether to load from Hugging Face (True) or local path (False).
        local_path: Local file/path to load if `use_hf=False`.
    """

    def __init__(self, no_image: bool = False, image: Optional[str] = None,
                 split: str = "train", use_hf: bool = True, local_path: Optional[str] = None):
        super().__init__(no_image, image)
        self.split = split
        self.use_hf = use_hf
        self.local_path = local_path

    def _choose_text(self, item: dict) -> str:
        # Common candidates for text/prompt fields
        for key in ("text", "prompt", "input", "query", "question", "source", "context"):
            if key in item and item.get(key) is not None:
                return str(item.get(key))
        # Fall back to the first string-like field
        for k, v in item.items():
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def _choose_label(self, raw: Any) -> int:
        # Normalize label to 0/1. Heuristics: treat negatives/untrusted as 0.
        if raw is None:
            return 1
        if isinstance(raw, int):
            return int(raw)
        s = str(raw).strip().lower()
        if s in ("0", "no", "false", "untrust", "untrusted", "untrustworthy", "unreliable", "un"):
            return 0
        if s in ("1", "yes", "true", "trust", "trusted", "trustworthy", "reliable"):
            return 1
        # fallback: consider labels containing 'un' as negative
        if "un" in s and len(s) > 2:
            return 0
        return 1

    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MultiTrust dataset and return prompts, labels, images, types."""
        if self.use_hf:
            try:
                ds = hf_load_dataset("thu-ml/MultiTrust")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load MultiTrust from Hugging Face: {e}. "
                    "Install the `datasets` library with `pip install datasets` and try again."
                )

            # The HF loader may return a dict of splits or a single Dataset
            if isinstance(ds, dict):
                if self.split in ds:
                    ds = ds[self.split]
                else:
                    # Try 'train' as fallback
                    ds = ds.get("train") or next(iter(ds.values()))

            prompts: List[str] = []
            labels: List[int] = []
            image_paths: List[Optional[str]] = []
            types: List[Optional[str]] = []

            for item in ds:
                # item is a dict-like mapping
                prompt = self._choose_text(item)
                prompts.append(prompt)

                # Image handling
                if not self.no_image and self.image is None:
                    img = item.get("image") or item.get("image_path") or item.get("image_url")
                    image_paths.append(img)
                else:
                    image_paths.append(self._get_image_path())

                # Labels: try common names
                raw_label = item.get("label") or item.get("labels") or item.get("trust") or item.get("label_id")
                labels.append(self._choose_label(raw_label))

                # Type or metadata
                type_val = item.get("type") or item.get("method") or item.get("policy") or None
                types.append(type_val)

        else:
            # Local loading fallback (CSV/JSONL). Keep implementation minimal.
            import os
            import pandas as pd
            from pathlib import Path

            if self.local_path is None:
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                self.local_path = project_root / "dataset" / "multitrust.csv"

            if not os.path.exists(self.local_path):
                raise FileNotFoundError(f"Local MultiTrust file not found: {self.local_path}")

            df = pd.read_csv(self.local_path)

            prompts = []
            labels = []
            image_paths = []
            types = []

            for _, row in df.iterrows():
                item = row.to_dict()
                prompts.append(self._choose_text(item))
                raw_label = item.get("label") or item.get("labels") or item.get("trust") or item.get("label_id")
                labels.append(self._choose_label(raw_label))
                if not self.no_image and self.image is None:
                    img = item.get("image") or item.get("image_path") or item.get("image_url")
                    image_paths.append(img)
                else:
                    image_paths.append(self._get_image_path())
                types.append(item.get("type") or None)

        return prompts, labels, image_paths, types
