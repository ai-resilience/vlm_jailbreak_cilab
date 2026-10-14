"""Input processing utilities for different VLM models."""
from PIL import Image
from typing import Tuple, Optional, Any, Dict, List
import sys
import os

# Add paths for DeepSeek utilities
# Add src/models to path so we can import deepseek_vl modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
sys.path.append(os.path.join(_project_root, "models"))

def build_template(model_name: str, img: Optional[str], prompt: str, system_prompt: Optional[str] = None) -> List[Dict]:
    """Build conversation template for model.
    
    Args:
        model_name: Name of the model
        img: Path to image (or None)
        prompt: Text prompt
        system_prompt: System prompt (optional, default: None)
        
    Returns:
        Conversation template
    """
    user_content = []
    
    if img is not None:
        user_content.append({"type": "image", "image": img})
    
    user_content.append({"type": "text", "text": prompt})
    
    if model_name in ["llava", "llava_next", "qwen", "gemma", "kimi"]:
        messages = []
        if system_prompt is not None:
            system_prompt = [{"type": "text", "text": system_prompt}]
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    if model_name == "phi":
        # Phi: content must be string; image handled in build_prompt
        messages = f"<|user|>{prompt}<|end|><|assistant|>" if img is None else f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
        return messages
    
    elif model_name == "intern":
        # For intern, system_prompt will be handled in build_prompt by setting model.conv_template.system_message
        if img is None:
            return [None, f"{prompt}"]
        else:
            return [f"{img}", f"<image>\n{prompt}"]
    
    elif model_name == "deepseek":
        # For deepseek, add system role if system_prompt is provided
        messages = []
        if system_prompt is not None and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if img is None:
            messages.extend([
                {"role": "User", "content": f"{prompt}."},
                {"role": "Assistant", "content": ""}
            ])
        else:
            messages.extend([
                {"role": "User", "content": f"<image_placeholder>{prompt}.", "images": [img]},
                {"role": "Assistant", "content": ""}
            ])
        return messages
    
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
    prompt: str,
    system_prompt: Optional[str] = None
) -> Tuple[Any, Optional[Any]]:
    """Build model inputs from prompt and image.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        img: Path to image
        prompt: Text prompt
        system_prompt: System prompt (optional, default: None)
        
    Returns:
        Tuple of (inputs, attention_mask)
    """
    message = build_template(model_name, img, prompt, system_prompt)
    
    if model_name == "deepseek":
        from ..models.deepseek_vl.utils.io import load_pil_images
        
        # Extract system_prompt from message if present
        system_prompt_value = ""
        if message and isinstance(message[0], dict) and message[0].get("role") == "system":
            system_prompt_value = message[0].get("content", "")
            message = message[1:]  # Remove system message from conversations
        
        # Temporarily set processor's system_prompt if provided
        original_system_prompt = processor.system_prompt
        if system_prompt_value:
            processor.system_prompt = system_prompt_value
        
        try:
            if img is None:
                text = processor(conversations=message, images=[], force_batchify=True).to(model.device)
            else:
                image = load_pil_images(message)
                text = processor(conversations=message, images=image, force_batchify=True).to(model.device)
        finally:
            # Restore original system_prompt
            processor.system_prompt = original_system_prompt
        
        input_embeds = model.prepare_inputs_embeds(**text)
        return input_embeds, text.attention_mask
    
    elif model_name == "deepseek2":
        from ..models.deepseek_vl2.utils.io import load_pil_images
        
        # Use provided system_prompt or default to empty string
        system_prompt_value = system_prompt if system_prompt is not None else ""
        
        if img is None:
            text = processor(conversations=message, images=[], force_batchify=True, system_prompt=system_prompt_value).to(model.device)
        else:
            image = load_pil_images(message)
            text = processor(conversations=message, images=image, force_batchify=True, system_prompt=system_prompt_value).to(model.device)
        
        input_embeds = model.prepare_inputs_embeds(**text)
        return input_embeds, text
    
    elif model_name == "qwen":
        from ..models.qwen.vision_process import process_vision_info
        
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
        # For intern, system_prompt will be handled in response.py by setting model.conv_template.system_message
        # Just return the message as is (system_prompt handling is done in generate_response)
        return message, None

    elif model_name == "kimi":
        text = processor.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt")
        if img is not None:
            image = Image.open(img)
        else:
            image = None
        inputs = processor(text=[text], images=image, padding=True, return_tensors="pt")
        return inputs, None
    
    elif model_name == "phi":
        # Phi: build string prompt, pass images separately
        if img is not None:
            image = Image.open(img)
        else:
            image = None
        inputs = processor(text=message, images=image, return_tensors='pt')

        return inputs, None
    
    else:  # llava, llava_next, qwen, gemma
        inputs = processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        return inputs, None

