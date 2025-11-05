"""Model implementations for various Vision-Language Models."""
from .base import BaseVLM
from .llava import LlavaModel, LlavaNextModel
from .qwen import QwenModel
from .intern import InternModel
from .deepseek_vl.deepseek import DeepSeekModel
from .deepseek_vl2.deepseek import DeepSeek2Model
from typing import Tuple, Any


def load_model(model_name: str, model_path: str = None) -> Tuple[Any, Any, Any]:
    """Factory function to load models by name.
    
    Args:
        model_name: Name of the model ('llava', 'llava_next', 'qwen', 'intern', 'deepseek', 'deepseek2')
        model_path: Optional custom model path
        
    Returns:
        Tuple of (model, processor, tokenizer)
    """
    model_map = {
        'llava': LlavaModel,
        'llava_next': LlavaNextModel,
        'qwen': QwenModel,
        'intern': InternModel,
        'deepseek': DeepSeekModel,
        'deepseek2': DeepSeek2Model,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    model_class = model_map[model_name]
    
    if model_path:
        model_obj = model_class(model_path)
    else:
        model_obj = model_class()
    
    return model_obj.load()


__all__ = [
    'BaseVLM',
    'LlavaModel',
    'LlavaNextModel',
    'QwenModel',
    'InternModel',
    'DeepSeekModel',
    'DeepSeek2Model',
    'load_model',
]

