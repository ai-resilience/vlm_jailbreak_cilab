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
        # Setup local transformers path for this load operation
        from pathlib import Path
        _deepseek_vl2_dir = Path(__file__).parent
        _transformers_path = _deepseek_vl2_dir / "transformers_4_38_2"
        _local_transformers_path = str(_transformers_path / "src")
        
        _path_added = False
        if _transformers_path.exists() and (_transformers_path / "src" / "transformers" / "__init__.py").exists():
            if _local_transformers_path not in sys.path:
                sys.path.insert(0, _local_transformers_path)
                _path_added = True
        
        try:
            from transformers import AutoModelForCausalLM
            from .models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        finally:
            # Remove local path after imports to avoid affecting other modules
            if _path_added and _local_transformers_path in sys.path:
                sys.path.remove(_local_transformers_path)
        
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

