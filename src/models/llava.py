"""LLaVA model implementations."""
import torch
from .base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional


class LlavaModel(BaseVLM):
    """LLaVA 1.5 model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "llava"
        super().__init__(model_path)
        
    def load(self, attn_implementation: Optional[str] = None) -> Tuple[Any, Any, Any]:
        """Load LLaVA 1.5 model.
        
        Args:
            attn_implementation: Attention implementation to use. 
                "eager" for standard attention (returns attention weights),
                "flash_attention_2" for Flash Attention (faster but no attention weights).
                Default: None (uses model default, which may be flash_attention_2)
        """
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        
        # Set attention implementation if provided
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs
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
        
    def load(self, attn_implementation: Optional[str] = None) -> Tuple[Any, Any, Any]:
        """Load LLaVA-NeXT model.
        
        Args:
            attn_implementation: Attention implementation to use. 
                "eager" for standard attention (returns attention weights),
                "flash_attention_2" for Flash Attention (faster but no attention weights).
                Default: None (uses model default, which may be flash_attention_2)
        """
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # Set attention implementation if provided
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs
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

