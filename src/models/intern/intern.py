"""InternVL model implementation."""
import torch
from ..base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional


class InternModel(BaseVLM):
    """InternVL3 model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "intern"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load InternVL model."""
        from transformers import AutoModel, AutoTokenizer
        
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
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)




