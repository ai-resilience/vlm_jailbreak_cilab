"""InternVL model implementation."""
import torch
from transformers import AutoModel, AutoTokenizer
from .base import BaseVLM
from typing import Tuple, Any


class InternModel(BaseVLM):
    """InternVL3 model wrapper."""
    
    def __init__(self, model_path: str = '/mnt/server14_hard1/kihyun/InternVL3-8B'):
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load InternVL model."""
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            use_flash_attn=True, 
            trust_remote_code=True
        ).eval().cuda()
        self.processor = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer = self.processor
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return self.model.config.llm_config.num_hidden_layers
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return self.model.language_model.norm

