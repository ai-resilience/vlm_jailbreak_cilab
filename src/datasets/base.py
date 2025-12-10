"""Base dataset loader interface."""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from dataclasses import dataclass
import os


@dataclass
class DatasetSample:
    """Single dataset sample."""
    prompt: str
    label: int
    image_path: Optional[str] = None
    type: Optional[str] = None


class BaseDataset(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, no_image: bool = True, image: Optional[str] = None):
        self.no_image = no_image
        self.image = image
        
    @abstractmethod
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load dataset.
        
        Returns:
            Tuple of (prompts, labels, image_paths, types)
        """
        pass
    
    def _get_image_path(self, num: int = 448) -> Optional[str]:
        """Get image path based on configuration."""
        if self.no_image:
            return None
        
        if os.path.exists(self.image):
            return self.image
        elif self.image == "blank":
            return f"dataset/blank/blank_{num}.png"
        elif self.image == "white":
            return f"dataset/white/white_{num}.png"
        elif self.image == "panda":
            return "dataset/clean.jpeg"
        elif self.image == "noise":
            return f"dataset/noise/noise_{num}.jpg"
        elif self.image == "res":
            return f"dataset/noise_{num}.png"
        elif self.image == "nsfw":
            return f"dataset/nsfw/temp_image_{num}.jpg"
        else:
            raise ValueError(f"Invalid image type: {self.image}")

