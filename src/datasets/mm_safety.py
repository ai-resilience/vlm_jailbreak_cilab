"""MM-SafetyBench dataset loaders."""
from typing import Tuple, List, Optional
from datasets import load_dataset as load_dataset_
from .base import BaseDataset


class MMTextDataset(BaseDataset):
    """MM-SafetyBench Text-only dataset."""
    
    def __init__(self, no_image: bool = False, image: Optional[str] = None):
        super().__init__(no_image, image)
        self.subsets = [
            'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
            'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
            'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
            'Privacy_Violence', 'Sex'
        ]
    
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
        # self.subsets = [
        #     'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
        #     'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
        #     'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
        #     'Privacy_Violence', 'Sex'
        # ]
        self.subsets = ['HateSpeech']
    
    def load(self) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
        """Load MM-SafetyBench TYPO dataset."""
        all_prompts, all_imgs, all_labels = [], [], []
        
        for subset in self.subsets:
            ds = load_dataset_("PKU-Alignment/MM-SafetyBench", subset, split="TYPO")
            all_prompts.extend(ds["question"])
            all_imgs.extend(ds["image"])
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
        self.subsets = [
            'EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision',
            'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion',
            'Malware_Generation', 'Physical_Harm', 'Political_Lobbying',
            'Privacy_Violence', 'Sex'
        ]
    
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

