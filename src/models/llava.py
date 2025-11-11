"""LLaVA model implementations."""
import torch
from .base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional


class LlavaModel(BaseVLM):
    """LLaVA 1.5 model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "llava"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load LLaVA 1.5 model."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
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
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)


class LlavaNextModel(BaseVLM):
    """LLaVA-NeXT model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "llava_next"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load LLaVA-NeXT model."""
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
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
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)

