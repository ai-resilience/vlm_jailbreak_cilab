"""Kimi-VL model implementation."""
import torch
from ..base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional


class KimiVLModel(BaseVLM):
    """Kimi-VL model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "kimi"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load Kimi-VL model."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)

