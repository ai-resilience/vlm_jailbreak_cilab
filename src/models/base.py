"""Base model interface for Vision-Language Models."""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self) -> Tuple[Any, Any, Any]:
        """Load model, processor, and tokenizer.
        
        Returns:
            Tuple of (model, processor, tokenizer)
        """
        pass
    
    @abstractmethod
    def get_num_layers(self) -> int:
        """Get number of hidden layers in the model."""
        pass
    
    @abstractmethod
    def get_norm_layer(self) -> Any:
        """Get the normalization layer of the model."""
        pass
    
    def get_device(self) -> str:
        """Get the device where the model is loaded."""
        if self.model is None:
            raise RuntimeError("Model not loaded yet")
        return str(self.model.device)

