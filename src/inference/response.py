"""Response generation utilities."""
import torch
from typing import Optional, Any
from .processor import build_prompt
import sys
import os

sys.path.append(os.path.abspath("utils/model/InternVL3"))


def generate_response(
    model: Any,
    processor: Any,
    model_name: str,
    prompt: str,
    img: Optional[str] = None,
    max_new_tokens: int = 128
) -> str:
    """Generate response from VLM.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        prompt: Text prompt
        img: Path to image (or None)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
    """
    inputs, attention_mask = build_prompt(model, processor, model_name, img, prompt)
    
    if model_name == "intern":
        from ..models.intern import load_image, generation_config
        
        prompt_text = inputs[1]
        image_path = inputs[0]
        
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        else:
            pixel_values = None
        
        with torch.no_grad():
            response = model.chat(processor, pixel_values, prompt_text, generation_config)
        
        return response
    
    elif model_name in ["deepseek", "deepseek2"]:
        inputs.to(model.device, dtype=torch.bfloat16)
        attention_mask = attention_mask.to(model.device)
        
        language_model = model.language_model if model_name == "deepseek" else model.language
        
        with torch.no_grad():
            out = language_model.generate(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
            out = out[0]
        
        response = processor.tokenizer.decode(out, skip_special_tokens=True)
        return response
    
    else:  # llava, llava_next, qwen, gemma
        inputs.to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out = out[0][input_len:]
        
        response = processor.tokenizer.decode(out, skip_special_tokens=True)
        return response

