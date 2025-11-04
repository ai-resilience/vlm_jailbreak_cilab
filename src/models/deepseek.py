"""DeepSeek-VL model implementations."""
import sys
import os
import torch
from transformers import AutoModelForCausalLM
from .base import BaseVLM
from typing import Tuple, Any

# Add DeepSeek paths
sys.path.append(os.path.abspath("utils/model/DeepSeek_VL"))
sys.path.append(os.path.abspath("utils/model/DeepSeek_VL2"))


class DeepSeekModel(BaseVLM):
    """DeepSeek-VL v1 model wrapper."""
    
    def __init__(self, model_path: str = '/mnt/server14_hard1/kihyun/deepseek-vl-7b-chat'):
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load DeepSeek-VL v1 model."""
        from utils.model.DeepSeek_VL.deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
        
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
        return self.model.config.language_config.num_hidden_layers
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return self.model.language_model.model.norm


class DeepSeek2Model(BaseVLM):
    """DeepSeek-VL v2 model wrapper."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load DeepSeek-VL v2 model."""
        # Implementation depends on DeepSeek v2 API
        # This is a placeholder
        raise NotImplementedError("DeepSeek v2 model loading needs to be implemented")
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return self.model.config.language_config.num_hidden_layers
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return self.model.language.model.norm

