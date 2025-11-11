"""AdvBench and other harmful benchmark datasets."""
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class AdvBenchDataset(BaseDataset):
    """AdvBench dataset."""
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load AdvBench dataset."""
        ds = hf_load_dataset("walledai/AdvBench")["train"]
        
        prompts = list(ds["prompt"])
        labels = [0] * len(prompts)  # All unsafe
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types


class HarmBenchDataset(BaseDataset):
    """HarmBench dataset."""
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load HarmBench dataset."""
        ds = hf_load_dataset("walledai/HarmBench", "standard")
        
        prompts = list(ds["train"]["prompt"])
        labels = [0] * len(prompts)  # All unsafe
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types


class SorryBenchDataset(BaseDataset):
    """SorryBench dataset."""
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load SorryBench dataset."""
        ds = hf_load_dataset("sorry-bench/sorry-bench-202503")["train"][:400]
        
        prompts = [turn[0] for turn in ds["turns"]]
        labels = [0] * len(prompts)  # All unsafe
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types

