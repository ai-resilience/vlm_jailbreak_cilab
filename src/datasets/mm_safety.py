"""MM-SafetyBench dataset loaders."""
from typing import Tuple, List, Optional
from datasets import load_dataset as load_dataset_
from PIL import Image
from .base import BaseDataset


class MMTextDataset(BaseDataset):
    """MM-SafetyBench Text-only dataset."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None):
        super().__init__(no_image, image)
        # self.subsets = [
        #     'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
        #     'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
        #     'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
        #     'Privacy_Violence', 'Sex'
        # ]
        self.subsets = ['HateSpeech']
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MM-SafetyBench Text-only dataset."""
        all_prompts, all_imgs, all_labels = [], [], []
        
        for subset in self.subsets:
            ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="Text_only")
            all_prompts.extend(ds["question"])
            all_imgs.extend(ds["image"])
            all_labels.extend([0] * len(ds["question"]))
        
        prompts = all_prompts
        imgs = all_imgs
        labels = all_labels
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types


class MMTypoDataset(BaseDataset):
    """MM-SafetyBench TYPO dataset."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None):
        super().__init__(no_image, image)
        self.subsets = [
            'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
            'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
            'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
            'Privacy_Violence', 'Sex'
        ]
        # self.subsets = ['HateSpeech']
    
    def _pad_image_to_height(self, img: Image.Image, target_height: int = 1024) -> Image.Image:
        """Pad image to target height with white pixels if needed.
        
        Args:
            img: PIL Image object
            target_height: Target height (default: 1024)
            
        Returns:
            Padded PIL Image with height = target_height
        """
        if not isinstance(img, Image.Image):
            return img
        
        width, height = img.size
        
        # If height is already >= target_height, return as is
        if height >= target_height:
            return img
        
        # Create a white background image with target dimensions
        padded_img = Image.new('RGB', (width, target_height), color='white')
        
        # Paste the original image at the top
        padded_img.paste(img, (0, 0))
        
        return padded_img
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MM-SafetyBench TYPO dataset."""
        all_prompts, all_imgs, all_labels = [], [], []
        
        for subset in self.subsets:
            ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="TYPO")
            all_prompts.extend(ds["question"])
            
            # Process images to pad height to 1024
            processed_imgs = []
            for img in ds["image"]:
                padded_img = self._pad_image_to_height(img, target_height=1024)
                processed_imgs.append(padded_img)
            
            all_imgs.extend(processed_imgs)
            all_labels.extend([0] * len(ds["question"]))
        
        prompts = all_prompts
        imgs = all_imgs
        labels = all_labels
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types


class MMSDTypoDataset(BaseDataset):
    """MM-SafetyBench SD_TYPO dataset."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None):
        super().__init__(no_image, image)
        # self.subsets = [
        #     'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
        #     'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
        #     'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
        #     'Privacy_Violence', 'Sex'
        # ]
        self.subsets = ['HateSpeech']
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MM-SafetyBench SD_TYPO dataset."""
        all_prompts, all_imgs, all_labels = [], [], []
        
        for subset in self.subsets:
            ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="SD_TYPO")
            all_prompts.extend(ds["question"])
            all_imgs.extend(ds["image"])
            all_labels.extend([0] * len(ds["question"]))
        
        prompts = all_prompts
        imgs = all_imgs
        labels = all_labels
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types

class MMSDDataset(BaseDataset):
    """MM-SafetyBench SD dataset."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None):
        super().__init__(no_image, image)
        # self.subsets = [
        #     'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
        #     'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
        #     'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
        #     'Privacy_Violence', 'Sex'
        # ]
        self.subsets = ['HateSpeech', 'Illegal_Activitiy']
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MM-SafetyBench SD dataset."""
        all_prompts, all_imgs, all_labels = [], [], []
        
        for subset in self.subsets:
            ds_1 = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="SD")
            ds_2 = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="Text_only")
            all_prompts.extend(ds_2["question"])
            all_imgs.extend(ds_1["image"])
            all_labels.extend([0] * len(ds_2["question"]))
        
        prompts = all_prompts
        imgs = all_imgs
        labels = all_labels
        types = [None] * len(prompts)
        
        return prompts, labels, imgs, types