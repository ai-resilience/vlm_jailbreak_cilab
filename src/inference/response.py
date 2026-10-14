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
    max_new_tokens: int = 128,
    system_prompt: Optional[str] = None
) -> str:
    """Generate response from VLM.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        prompt: Text prompt
        img: Path to image (or None)
        max_new_tokens: Maximum tokens to generate
        system_prompt: System prompt (optional, default: None)
        
    Returns:
        Generated text response
    """
    inputs, attention_mask = build_prompt(model, processor, model_name, img, prompt, system_prompt)
    
    if model_name == "intern":
        from ..models.intern.preprocess import load_image, generation_config
        
        prompt_text = inputs[1]
        image_path = inputs[0]
        
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        else:
            pixel_values = None
        
        # Set system_message in conv_template if system_prompt is provided
        original_system_message = None
        if system_prompt is not None:
            # Save original system_message
            original_system_message = model.conv_template.system_message
            # Set new system_message
            model.conv_template.system_message = system_prompt
        
        try:
            with torch.no_grad():
                response = model.chat(processor, pixel_values, prompt_text, generation_config)
        finally:
            # Restore original system_message if it was changed
            if original_system_message is not None:
                model.conv_template.system_message = original_system_message
        
        return response
    
    elif model_name in ["deepseek"]:
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

    elif model_name in ["deepseek2"]:
        prepare_inputs = attention_mask
        inputs_embeds, past_key_values = model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=512 # prefilling size
                    )
        
        out = model.generate(
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,)
        out = out[0][len(prepare_inputs.input_ids[0]):].cpu().tolist()
        
        response = processor.tokenizer.decode(out, skip_special_tokens=True)
        return response

    elif model_name == "phi":
        # Phi model: set adapter and generate
        inputs = inputs.to(model.device)
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                generation_config=model.generation_config,
                num_logits_to_keep=1,
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    else:  # llava, llava_next, qwen, gemma, kimi
        inputs.to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out = out[0][input_len:]
        
        response = processor.tokenizer.decode(out, skip_special_tokens=True)
        return response

