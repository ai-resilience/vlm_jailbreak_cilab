"""LLaVA model implementations."""
import torch
from transformers import (
    LlavaForConditionalGeneration, 
    LlavaNextForConditionalGeneration,
    AutoProcessor, 
    LlavaNextProcessor
)
from .base import BaseVLM
from typing import Tuple, Any


class LlavaModel(BaseVLM):
    """LLaVA 1.5 model wrapper."""
    
    def __init__(self, model_path: str = '/mnt/server11_hard4/kihyun/mil/llava-1.5-13b-hf'):
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load LLaVA 1.5 model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return self.model.config.text_config.num_hidden_layers
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return self.model.language_model.norm


class LlavaNextModel(BaseVLM):
    """LLaVA-NeXT model wrapper."""
    
    def __init__(self, model_path: str = "/mnt/server14_hard1/kihyun/llava-v1.6-mistral-7b-hf"):
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load LLaVA-NeXT model."""
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return self.model.config.text_config.num_hidden_layers
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return self.model.language_model.norm

