"""Qwen-VL model implementation."""
import torch
from .base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any


class QwenModel(BaseVLM):
    """Qwen2.5-VL model wrapper."""
    
    def __init__(self, model_path: str = '/mnt/server14_hard1/kihyun/Qwen2.5-VL-7B-Instruct'):
        super().__init__(model_path)
        self.model_name = "qwen"
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load Qwen-VL model."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)

