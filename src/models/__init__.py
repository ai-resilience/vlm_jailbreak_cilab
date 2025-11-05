"""Model implementations for various Vision-Language Models."""
import os
import yaml
from pathlib import Path
from .base import BaseVLM
from .llava import LlavaModel, LlavaNextModel
from .qwen.qwen import QwenModel
from .intern.intern import InternModel
from .deepseek_vl.deepseek import DeepSeekModel
from .deepseek_vl2.deepseek import DeepSeek2Model
from typing import Tuple, Any, Optional


def _load_config() -> dict:
    """Load model configurations from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('models', {})


def load_model(model_name: str, model_path: Optional[str] = None) -> Tuple[Any, Any, Any]:
    """Factory function to load models by name.
    
    Args:
        model_name: Name of the model ('llava', 'llava_next', 'qwen', 'intern', 'deepseek', 'deepseek2')
        model_path: Optional custom model path. If not provided, uses path from configs/models.yaml
        
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
    
    # Create model instance - it will automatically load from config if model_path is None
    model_obj = model_class(model_path)
    
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

