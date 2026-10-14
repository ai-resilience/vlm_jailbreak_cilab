"""StrongREJECT dataset loader."""
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class HadesDataset(BaseDataset):
    """Hades dataset for safety evaluation."""
    
    def __init__(self, no_image: bool = True, image: Optional[str] = None, filter_instruction: bool = True, no_typo: bool = False):
        super().__init__(no_image, image)
        self.filter_instruction = filter_instruction
        self.no_typo = no_typo
        
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load Hades dataset."""
        ds = hf_load_dataset("Monosail/HADES", split="test")
        
        prompts = []
        imgs = []
        for item in ds:
            inst = item["instruction"]
            img = item["image"].convert("RGB")
            if self.filter_instruction:
                if item["category"] == "behavior":
                    mask = "conduct this behavior on"
                else:
                    mask = f"this {item['category']}"
                new_inst = f"The bottom of the image shows a {item['category']}. " + inst.replace(item["keywords"], mask)
            else:
                new_inst = inst
            prompts.append(new_inst)
            
            if self.no_typo:
                # Crop top 1024x1024 (width is already 1024, height is longer)
                new_img = img.crop((0, 0, 1024, 1024))
                imgs.append(new_img)
            else:
                imgs.append(img)
                
        labels = [0] * len(prompts)  # All are unsafe
        types = [None] * len(prompts)
                    
        
        return prompts, labels, imgs, types

