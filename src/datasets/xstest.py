"""XSTest dataset loader."""
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class XSTestDataset(BaseDataset):
    """XSTest dataset for false positive evaluation."""
    
    def __init__(self, no_image: bool = True, image: Optional[str] = None, 
                 filter_safe: bool = False, filter_unsafe: bool = False):
        super().__init__(no_image, image)
        self.filter_safe = filter_safe
        self.filter_unsafe = filter_unsafe
        
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load XSTest dataset."""
        valid_types = [
            "homonyms", "contrast_homonyms",
            "figurative_language", "contrast_figurative_language",
            "safe_targets", "contrast_safe_targets",
            "safe_contexts", "contrast_safe_contexts"
        ]
        
        ds = hf_load_dataset("walledai/XSTest")
        ds = ds["test"].filter(lambda x: x["type"] in valid_types)
        
        # Apply filters
        if self.filter_safe:
            ds = ds.filter(lambda x: x["label"] == "safe")
        elif self.filter_unsafe:
            ds = ds.filter(lambda x: x["label"] == "unsafe")
        
        prompts = list(ds["prompt"])
        labels = [1 if l == "safe" else 0 for l in ds["label"]]
        types = list(ds["type"])
        
        image_path = self._get_image_path()
        imgs = [image_path] * len(prompts)
        
        return prompts, labels, imgs, types

