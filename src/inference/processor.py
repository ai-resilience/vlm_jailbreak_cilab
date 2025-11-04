"""Input processing utilities for different VLM models."""
from PIL import Image
from typing import Tuple, Optional, Any, Dict, List
import sys
import os

# Add paths for DeepSeek utilities
sys.path.append(os.path.abspath("utils/model/DeepSeek_VL"))
sys.path.append(os.path.abspath("utils/model/DeepSeek_VL2"))


def build_template(model_name: str, img: Optional[str], prompt: str) -> List[Dict]:
    """Build conversation template for model.
    
    Args:
        model_name: Name of the model
        img: Path to image (or None)
        prompt: Text prompt
        
    Returns:
        Conversation template
    """
    user_content = []
    
    if img is not None:
        user_content.append({"type": "image", "image": img})
    
    user_content.append({"type": "text", "text": prompt})
    
    if model_name in ["llava", "llava_next", "qwen", "gemma"]:
        return [{"role": "user", "content": user_content}]
    
    elif model_name == "intern":
        if img is None:
            return [f"{img}", f"{prompt}"]
        else:
            return [f"{img}", f"<image>\n{prompt}"]
    
    elif model_name == "deepseek":
        if img is None:
            return [
                {"role": "User", "content": f"{prompt}."},
                {"role": "Assistant", "content": ""}
            ]
        else:
            return [
                {"role": "User", "content": f"<image_placeholder>{prompt}.", "images": [img]},
                {"role": "Assistant", "content": ""}
            ]
    
    elif model_name == "deepseek2":
        if img is None:
            return [
                {"role": "<|User|>", "content": f"{prompt}"},
                {"role": "<|Assistant|>", "content": ""}
            ]
        else:
            return [
                {"role": "<|User|>", "content": f"<image>\n {prompt}", "images": [img]},
                {"role": "<|Assistant|>", "content": ""}
            ]
    
    raise ValueError(f"Unknown model: {model_name}")


def build_prompt(
    model: Any,
    processor: Any,
    model_name: str,
    img: Optional[str],
    prompt: str
) -> Tuple[Any, Optional[Any]]:
    """Build model inputs from prompt and image.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        img: Path to image
        prompt: Text prompt
        
    Returns:
        Tuple of (inputs, attention_mask)
    """
    message = build_template(model_name, img, prompt)
    
    if model_name == "deepseek":
        from utils.model.DeepSeek_VL.deepseek_vl.utils.io import load_pil_images
        
        if img is None:
            text = processor(conversations=message, images=[], force_batchify=True).to(model.device)
        else:
            image = load_pil_images(message)
            text = processor(conversations=message, images=image, force_batchify=True).to(model.device)
        
        input_embeds = model.prepare_inputs_embeds(**text)
        return input_embeds, text.attention_mask
    
    elif model_name == "deepseek2":
        from utils.model.DeepSeek_VL2.deepseek_vl2.utils.io import load_pil_images as load_pil_images_2
        
        if img is None:
            text = processor(conversations=message, images=[], force_batchify=True, system_prompt="").to(model.device)
        else:
            image = load_pil_images_2(message)
            text = processor(conversations=message, images=image, force_batchify=True, system_prompt="").to(model.device)
        
        input_embeds = model.prepare_inputs_embeds(**text)
        return input_embeds, text.attention_mask
    
    elif model_name == "qwen":
        from utils.model.Qwen_VL.qwen_vl_utils.src.qwen_vl_utils.vision_process import process_vision_info
        
        if img is None:
            inputs = processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            )
        else:
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(message)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        
        return inputs, None
    
    elif model_name == "intern":
        return message, None
    
    else:  # llava, llava_next, gemma
        inputs = processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        return inputs, None

