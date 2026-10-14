"""JailBreakV-28K dataset loader."""
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset


class JailBreakV28KDataset(BaseDataset):
    """JailBreakV-28K dataset for multimodal jailbreak evaluation."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None, 
                 split: str = "train", use_hf: bool = True, local_path: Optional[str] = None):
        """Initialize JailBreakV-28K dataset.
        
        Args:
            no_image: Whether to use text-only mode
            image: Image type to use (overrides dataset images if provided)
            split: Dataset split to load ('train' or 'test')
            use_hf: Whether to load from Hugging Face (True) or local CSV (False)
            local_path: Path to local CSV file if use_hf=False
        """
        super().__init__(no_image, image)
        self.split = split
        self.use_hf = use_hf
        self.local_path = local_path
        
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load JailBreakV-28K dataset.
        
        Returns:
            Tuple of (prompts, labels, image_paths, types)
        """
        if self.use_hf:
            # Load from Hugging Face
            try:
                ds = hf_load_dataset("JailbreakV-28K/JailBreakV-28k", split=self.split)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load JailBreakV-28K from Hugging Face: {e}. "
                    "Make sure you have 'datasets' library installed: pip install datasets"
                )
            
            # Extract data - adjust column names based on actual dataset structure
            # Common column names: 'prompt', 'query', 'text', 'image', 'image_path', 'label', 'type'
            prompts = []
            image_paths = []
            labels = []
            types = []
            
            for item in ds:
                # Try different possible column names for prompt
                prompt = item.get("prompt") or item.get("query") or item.get("text") or ""
                prompts.append(prompt)
                
                # Try different possible column names for image
                if not self.no_image and self.image is None:
                    # Use image from dataset if available
                    img = item.get("image") or item.get("image_path") or item.get("image_url")
                    image_paths.append(img)
                else:
                    # Use configured image or None
                    image_path = self._get_image_path()
                    image_paths.append(image_path)
                
                # Label: 0 for unsafe (jailbreak), 1 for safe
                # JailBreakV-28K contains jailbreak attempts, so all are unsafe (0)
                label = item.get("label", 0)
                if isinstance(label, str):
                    label = 0 if label.lower() in ["unsafe", "jailbreak", "harmful"] else 1
                labels.append(label)
                
                # Type: jailbreak method or safety policy
                type_val = item.get("type") or item.get("method") or item.get("policy") or None
                types.append(type_val)
                
        else:
            # Load from local CSV file
            import pandas as pd
            import os
            from pathlib import Path
            
            if self.local_path is None:
                # Default path: dataset/JailBreakV_28K.csv
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent
                self.local_path = project_root / "dataset" / "JailBreakV_28K.csv"
            
            if not os.path.exists(self.local_path):
                raise FileNotFoundError(
                    f"JailBreakV-28K dataset file not found: {self.local_path}. "
                    f"Please download the dataset or specify the correct path."
                )
            
            df = pd.read_csv(self.local_path)
            
            prompts = []
            image_paths = []
            labels = []
            types = []
            
            for _, row in df.iterrows():
                # Extract prompt
                prompt = row.get("prompt") or row.get("query") or row.get("text") or ""
                prompts.append(prompt)
                
                # Extract image path
                if not self.no_image and self.image is None:
                    img = row.get("image") or row.get("image_path") or row.get("image_url")
                    # If image is a relative path, make it absolute relative to dataset directory
                    if img and not os.path.isabs(img):
                        img = os.path.join(os.path.dirname(self.local_path), img)
                    image_paths.append(img)
                else:
                    image_path = self._get_image_path()
                    image_paths.append(image_path)
                
                # Label
                label = row.get("label", 0)
                if isinstance(label, str):
                    label = 0 if label.lower() in ["unsafe", "jailbreak", "harmful"] else 1
                labels.append(label)
                
                # Type
                type_val = row.get("type") or row.get("method") or row.get("policy") or None
                types.append(type_val)
        
        return prompts, labels, image_paths, types

