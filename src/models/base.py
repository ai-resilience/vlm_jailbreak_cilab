"""Base model interface for Vision-Language Models."""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional


def find_norm(model: Any) -> Any:
    """Find the normalization layer in the model by trying multiple paths.
    Args:
        model: The model object
    Returns:
        The normalization layer
    Raises:
        ValueError: If no norm layer is found
    """
    for path in [
        "language_model.norm",
        "language_model.model.norm",
        "language.model.norm",
    ]:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"✅ Found norm at: model.{path}")
            return obj
        except AttributeError:
            continue
    raise ValueError("❌ Could not find any norm layer.")


def find_num_hidden_layers(model: Any) -> int:
    """Find the number of hidden layers by trying multiple paths.
    Args:
        model: The model object
    Returns:
        Number of hidden layers
    Raises:
        ValueError: If no num_hidden_layers is found
    """
    for path in [
        "config.language_config.num_hidden_layers",
        "config.llm_config.num_hidden_layers",
        "config.text_config.num_hidden_layers",
    ]:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"✅ Found num_hidden_layers at: model.{path}")
            return obj
        except AttributeError:
            continue
    raise ValueError("❌ Could not find any num_hidden_layers.")


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_name = None  # Should be set by subclasses
        
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

