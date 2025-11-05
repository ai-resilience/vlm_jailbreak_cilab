"""DeepSeek-VL model implementations."""
import sys
import os
import torch
from ..base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional

class DeepSeek2Model(BaseVLM):
    """DeepSeek-VL v2 model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "deepseek2"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load DeepSeek-VL v2 model."""
        from transformers import AutoModelForCausalLM
        from .models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)

