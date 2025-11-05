"""DeepSeek-VL model implementations."""
import sys
import os
import torch
from ..base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional

class DeepSeekModel(BaseVLM):
    """DeepSeek-VL v1 model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "deepseek"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load DeepSeek-VL v1 model."""
        from transformers import AutoModelForCausalLM
        from .models import MultiModalityCausalLM, VLChatProcessor
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        self.processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)


