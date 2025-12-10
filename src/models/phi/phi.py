"""Phi-4 multimodal model implementation."""
import torch
from ..base import BaseVLM, find_norm, find_num_hidden_layers
from typing import Tuple, Any, Optional


class PhiModel(BaseVLM):
    """Phi-4 multimodal model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_name = "phi"
        super().__init__(model_path)
        
    def load(self) -> Tuple[Any, Any, Any]:
        """Load Phi-4 multimodal model."""
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        self.generation_config = GenerationConfig.from_pretrained(self.model_path, 'generation_config.json')
        
        return self.model, self.processor, self.tokenizer
    
    def get_num_layers(self) -> int:
        """Get number of hidden layers."""
        return find_num_hidden_layers(self.model)
    
    def get_norm_layer(self) -> Any:
        """Get normalization layer."""
        return find_norm(self.model)
