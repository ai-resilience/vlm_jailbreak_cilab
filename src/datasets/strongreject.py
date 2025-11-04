"""StrongREJECT dataset loader."""
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class StrongREJECTDataset(BaseDataset):
    """StrongREJECT dataset for safety evaluation."""
    
    def __init__(self, no_image: bool = True, image: Optional[str] = None):
        super().__init__(no_image, image)
        
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load StrongREJECT dataset."""
        ds = hf_load_dataset("walledai/StrongREJECT", split="train")
        
        prompts = list(ds["prompt"])
        labels = [0] * len(prompts)  # All are unsafe
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types

