"""LLMSafeguard dataset loader."""
import json
import os
from typing import Tuple, List, Optional
from pathlib import Path
from .base import BaseDataset


class LLMSafeguardDataset(BaseDataset):
    """LLMSafeguard dataset for safety evaluation."""
    
    def __init__(self, no_image: bool = True, image: Optional[str] = None):
        super().__init__(no_image, image)
        # Find project root (2 levels up from src/datasets/llmsafeguard.py)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/datasets -> src -> project_root
        self.data_path = project_root / "dataset" / "llmsafeguard_dataset.jsonl"
        
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load LLMSafeguard dataset from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LLMSafeguard dataset file not found: {self.data_path}. "
                f"Please ensure the file exists at dataset/llmsafeguard_dataset.jsonl"
            )
        
        prompts = []
        labels = []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    prompts.append(data["prompt"])
                    labels.append(data["label"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping invalid line in {self.data_path}: {e}")
                    continue
        
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types

