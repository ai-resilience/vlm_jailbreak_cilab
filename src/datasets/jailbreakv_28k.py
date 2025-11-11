"""JailBreakV-28K dataset loader."""
import os
from typing import Tuple, List, Optional
from datasets import load_dataset as hf_load_dataset
from .base import BaseDataset

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class JailBreakV28KDataset(BaseDataset):
    """JailBreakV-28K dataset for multimodal jailbreak evaluation.
    
    This dataset contains 28,000 jailbreak text-image pairs including:
    - 20,000 text-based LLM transfer jailbreak attacks
    - 8,000 image-based MLLM jailbreak attacks
    
    Sources:
    - Hugging Face: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k
    - GitHub: https://github.com/SaFoLab-WISC/JailBreakV_28K
    """
    
    def __init__(
        self, 
        no_image: bool = False, 
        image: Optional[str] = None,
        source: str = "huggingface",
        data_path: Optional[str] = None
    ):
        """
        Args:
            no_image: Whether to use text-only mode
            image: Image type to use (if no_image=False and dataset image not used)
            source: Data source - "huggingface" or "github" or "local"
            data_path: Local path to CSV file (if source="local")
        """
        super().__init__(no_image, image)
        self.source = source
        self.data_path = data_path
        
    def _load_from_huggingface(self):
        """Load dataset from Hugging Face."""
        if pd is None:
            raise ImportError("pandas is required to load JailBreakV-28K dataset. Install with: pip install pandas")
        
        try:
            # Try loading as a dataset first
            ds = hf_load_dataset("JailbreakV-28K/JailBreakV-28k", split="train")
            # Convert to pandas DataFrame
            df = ds.to_pandas()
            return df
        except Exception as e:
            # If dataset format doesn't work, try loading CSV directly
            if hf_hub_download is None:
                raise ImportError(
                    "huggingface_hub is required for CSV download. "
                    "Install with: pip install huggingface_hub"
                )
            try:
                csv_path = hf_hub_download(
                    repo_id="JailbreakV-28K/JailBreakV-28k",
                    filename="JailBreakV_28K.csv",
                    repo_type="dataset"
                )
                df = pd.read_csv(csv_path)
                return df
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load from Hugging Face: {e}. "
                    f"Also tried CSV download: {e2}"
                )
    
    def _load_from_github(self):
        """Load dataset from GitHub repository."""
        if pd is None:
            raise ImportError("pandas is required to load JailBreakV-28K dataset. Install with: pip install pandas")
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for GitHub download. Install with: pip install requests")
        
        from io import StringIO
        
        # GitHub raw content URL for the CSV file
        github_url = (
            "https://raw.githubusercontent.com/SaFoLab-WISC/JailBreakV_28K/V0.2/"
            "JailBreakV_28K/JailBreakV_28K.csv"
        )
        
        try:
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from string
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load from GitHub: {e}")
    
    def _load_from_local(self):
        """Load dataset from local file."""
        if pd is None:
            raise ImportError("pandas is required to load JailBreakV-28K dataset. Install with: pip install pandas")
        
        if self.data_path is None:
            raise ValueError("data_path must be provided when source='local'")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        return df
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load JailBreakV-28K dataset.
        
        Returns:
            Tuple of (prompts, labels, image_paths, types)
        """
        # Load dataset based on source
        if self.source == "huggingface":
            df = self._load_from_huggingface()
        elif self.source == "github":
            df = self._load_from_github()
        elif self.source == "local":
            df = self._load_from_local()
        else:
            raise ValueError(f"Unknown source: {self.source}. Use 'huggingface', 'github', or 'local'")
        
        # Extract data based on CSV structure
        # Note: Column names may vary, adjust based on actual CSV structure
        # Common column names: 'prompt', 'text', 'query', 'image', 'image_path', 'label', etc.
        
        # Try to find prompt/text column
        prompt_col = None
        for col in ['prompt', 'text', 'query', 'question', 'instruction']:
            if col in df.columns:
                prompt_col = col
                break
        
        if prompt_col is None:
            raise ValueError(
                f"Could not find prompt column in dataset. Available columns: {df.columns.tolist()}"
            )
        
        prompts = df[prompt_col].tolist()
        
        # Try to find image column
        image_col = None
        for col in ['image', 'image_path', 'image_url', 'img', 'img_path']:
            if col in df.columns:
                image_col = col
                break
        
        # Handle images
        if self.no_image:
            image_paths = [None] * len(prompts)
        elif image_col and image_col in df.columns:
            # Use images from dataset
            image_paths = df[image_col].tolist()
        else:
            # Fall back to default image handling
            image_path = self._get_image_path()
            image_paths = [image_path] * len(prompts)
        
        # Try to find label column
        label_col = None
        for col in ['label', 'is_unsafe', 'safety_label', 'target']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            # Convert labels to binary (0=unsafe, 1=safe)
            labels_raw = df[label_col].tolist()
            labels = []
            for lbl in labels_raw:
                if isinstance(lbl, bool):
                    labels.append(0 if lbl else 1)
                elif isinstance(lbl, str):
                    labels.append(0 if lbl.lower() in ['unsafe', 'harmful', '1', 'true'] else 1)
                elif isinstance(lbl, (int, float)):
                    labels.append(0 if lbl == 0 or lbl == 1 else 1)
                else:
                    labels.append(0)  # Default to unsafe
        else:
            # Default: all are unsafe (jailbreak attacks)
            labels = [0] * len(prompts)
        
        # Try to find type column
        type_col = None
        for col in ['type', 'attack_type', 'method', 'category']:
            if col in df.columns:
                type_col = col
                break
        
        if type_col and type_col in df.columns:
            types = df[type_col].tolist()
        else:
            types = [None] * len(prompts)
        
        return prompts, labels, image_paths, types

