"""Base model interface for Vision-Language Models."""
import yaml
from pathlib import Path
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
        "model.norm"
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
        "config.num_hidden_layers",
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


def find_lm_head(model: Any) -> Any:
    """Find the language model head (lm_head) by trying multiple paths.
    Args:
        model: The model object
    Returns:
        The language model head layer
    Raises:
        ValueError: If no lm_head is found
    """
    for path in [
        "lm_head",
        "language_model.lm_head",
        "language.lm_head",
    ]:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"✅ Found lm_head at: model.{path}")
            return obj
        except AttributeError:
            continue
    raise ValueError("❌ Could not find any lm_head.")


def find_layers(model: Any) -> Any:
    """Find the layers module in the model by trying multiple paths.
    Args:
        model: The model object
    Returns:
        The layers module (e.g., nn.ModuleList of transformer layers)
    Raises:
        ValueError: If no layers module is found
    """
    for path in [
        "language_model.layers",
        "language_model.model.layers",
        "language.layers",
        "language.model.layers",
    ]:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            # Check if it's actually a layers module (has __len__ and __getitem__)
            if hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
                print(f"✅ Found layers at: model.{path}")
                return obj
        except (AttributeError, TypeError):
            continue
    raise ValueError("❌ Could not find any layers module.")


def _load_model_config(model_name: str) -> Optional[str]:
    """Load model path from configs/models.yaml.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model path from config, or None if not found
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        models = config.get('models', {})
        if model_name in models:
            return models[model_name].get('path')
    except Exception:
        pass
    
    return None


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(self, model_path: Optional[str] = None):
        # If model_path not provided, try to load from config
        if model_path is None:
            if self.model_name:
                model_path = _load_model_config(self.model_name)
            if model_path is None:
                raise ValueError(
                    f"model_path must be provided or defined in configs/models.yaml for {self.__class__.__name__}"
                )
        
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        # model_name should be set by subclasses before calling super().__init__()
        
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

