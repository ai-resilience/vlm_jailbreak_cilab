"""Dataset loaders for various safety benchmarks."""
from .base import BaseDataset, DatasetSample
from .figstep import FigStepDataset
from .strongreject import StrongREJECTDataset
from .xstest import XSTestDataset
from .advbench import AdvBenchDataset, HarmBenchDataset, SorryBenchDataset
from .llmsafeguard import LLMSafeguardDataset
from .mm_safety import MMTextDataset, MMTypoDataset, MMSDTypoDataset
from .jailbreakv_28k import JailBreakV28KDataset
from typing import Tuple, List, Optional


def load_dataset(
    name: str, 
    no_image: bool = True, 
    image: Optional[str] = None,
    **kwargs
) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[str]]]:
    """Factory function to load datasets by name.
    
    Args:
        name: Dataset name
        no_image: Whether to use text-only mode
        image: Image type to use
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Tuple of (prompts, labels, image_paths, types)
    """
    dataset_map = {
        'Figstep': FigStepDataset,
        'StrongREJECT': StrongREJECTDataset,
        'XSTest': XSTestDataset,
        'XSTest_safe': lambda **kw: XSTestDataset(filter_safe=True, **kw),
        'XSTest_unsafe': lambda **kw: XSTestDataset(filter_unsafe=True, **kw),
        'AdvBench': AdvBenchDataset,
        'HarmBench': HarmBenchDataset,
        'SorryBench': SorryBenchDataset,
        'llmsafeguard': LLMSafeguardDataset,
        'mm_text': MMTextDataset,
        'mm_typo': MMTypoDataset,
        'mm_sd_typo': MMSDTypoDataset,
        'JailBreakV-28K': lambda **kw: JailBreakV28KDataset(source='huggingface', **kw),
        'JailBreakV-28K_hf': lambda **kw: JailBreakV28KDataset(source='huggingface', **kw),
        'JailBreakV-28K_github': lambda **kw: JailBreakV28KDataset(source='github', **kw),
        'JailBreakV-28K_local': lambda **kw: JailBreakV28KDataset(source='local', **kw),
    }
    
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[name]
    
    if callable(dataset_class) and not isinstance(dataset_class, type):
        # It's a lambda function
        dataset = dataset_class(no_image=no_image, image=image, **kwargs)
    else:
        dataset = dataset_class(no_image=no_image, image=image)
    
    return dataset.load()


__all__ = [
    'BaseDataset',
    'DatasetSample',
    'FigStepDataset',
    'StrongREJECTDataset',
    'XSTestDataset',
    'AdvBenchDataset',
    'HarmBenchDataset',
    'SorryBenchDataset',
    'LLMSafeguardDataset',
    'MMTextDataset',
    'MMTypoDataset',
    'MMSDTypoDataset',
    'JailBreakV28KDataset',
    'load_dataset',
]

