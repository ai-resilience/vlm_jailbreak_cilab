"""Hidden state extraction and analysis utilities."""
import torch
import numpy as np
from typing import List, Optional, Any


def extract_hidden_states(
    model: Any,
    processor: Any,
    model_name: str,
    prompt: str,
    image_path: Optional[str],
    build_prompt_fn: Any
) -> List[torch.Tensor]:
    """Extract hidden states from all layers.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        prompt: Text prompt
        image_path: Path to image (or None for text-only)
        build_prompt_fn: Function to build model inputs
        
    Returns:
        List of hidden states per layer [layer_idx][batch, seq_len, hidden_dim]
    """
    inputs, attention_mask = build_prompt_fn(model, processor, model_name, image_path, prompt)

    
    if model_name == "deepseek2":
        inputs.to(model.device)
        attention_mask.to(model.device)
        with torch.no_grad():
            out = model.language(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )

    elif model_name == "intern":
        from ..models.intern.preprocess import load_image
        
        prompt_text = inputs[1]
        image_path = inputs[0]
        
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
        else:
            pixel_values = None

        # Prepare template and query (similar to model.chat logic)
        if pixel_values is not None and '<image>' not in prompt_text:
            prompt_text = '<image>\n' + prompt_text
        
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        
        # Set img_context_token_id
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        model.img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        # Build template using model's conv_template
        template = model.conv_template.copy()
        template.system_message = model.system_message
        template.append_message(template.roles[0], prompt_text)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        # Insert image tokens
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        # Tokenize
        model_inputs = processor(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(model.device)
        attention_mask = model_inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        
        # Create image_flags for InternVL3 forward method
        # In forward(), image_flags is used as: vit_embeds[image_flags == 1]
        # where vit_embeds has shape [vit_batch_size, num_tokens, hidden_size]
        # 
        # IMPORTANT: image_flags is NOT about text sequence token positions!
        # It's about which images in pixel_values are actually used.
        # After squeeze(-1), image_flags should have shape [vit_batch_size, ...]
        # where the first dimension indicates which images are used (1) or not (0).
        #
        # For our use case, all images in pixel_values are used, so we create
        # image_flags with shape [vit_batch_size, 1, 1] where all values are 1.
        if pixel_values is not None:
            vit_batch_size = pixel_values.shape[0]
            text_batch_size, seq_len = input_ids.shape
            
            # image_flags should indicate which images are used
            # After squeeze(-1) in forward(), it should become [vit_batch_size] (1D)
            # so that vit_embeds[image_flags == 1] can index the first dimension correctly
            # Since all images are used, create all-ones with shape [vit_batch_size, 1]
            # After squeeze(-1): [vit_batch_size] -> first dim used for indexing vit_embeds
            image_flags = torch.ones(vit_batch_size, 1, dtype=torch.long, device=input_ids.device)
        else:
            # Text-only: create zeros with shape matching input_ids
            image_flags = torch.zeros_like(input_ids).unsqueeze(-1)
        
        # Call model.forward directly to get hidden states
        with torch.no_grad():
            if pixel_values is not None:
                # Image + text case: use model.forward
                out = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                # Text-only case: call language_model directly
                # No image tokens should be in input_ids for text-only
                out = model.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

    else:
        inputs.to(model.device)
        with torch.no_grad():
            out = model(**inputs, return_dict=True, output_hidden_states=True)
    
    return out.hidden_states


def compute_cosine_similarity(vectors_1: List[np.ndarray], vectors_2: List[np.ndarray]) -> List[float]:
    """Compute cosine similarity between two sets of vectors.
    
    Args:
        vectors_1: First set of vectors
        vectors_2: Second set of vectors
        
    Returns:
        List of cosine similarities
    """
    cosine_sims = []
    for vec_1, vec_2 in zip(vectors_1, vectors_2):
        vec_1 = torch.tensor(vec_1, dtype=torch.float32)
        vec_2 = torch.tensor(vec_2, dtype=torch.float32)
        cos_sim = torch.cosine_similarity(vec_1, vec_2, dim=0)
        cosine_sims.append(cos_sim.item())
    return cosine_sims


def extract_token_hidden_states(
    hidden_states: List[torch.Tensor],
    norm_layer: Any,
    token_index: int = -1,
    num_layers: Optional[int] = None
) -> List[np.ndarray]:
    """Extract hidden states for a specific token position from all layers.
    
    Args:
        hidden_states: Hidden states from model [layer][batch, seq_len, dim]
        norm_layer: Normalization layer
        token_index: Token position to extract (-1 for last token)
        num_layers: Number of layers (if None, use all)
        
    Returns:
        List of extracted vectors per layer
    """
    if num_layers is None:
        num_layers = len(hidden_states)
    
    vectors = []
    for idx in range(num_layers):
        vec = hidden_states[idx][:, token_index, :]
        normed_vec = norm_layer(vec).squeeze(0).detach().cpu().float().numpy()
        vectors.append(normed_vec)
    
    return vectors


def save_hidden_states(
    vectors_all_layers: List[List[np.ndarray]],
    save_path: str
) -> None:
    """Save hidden states to compressed numpy file.
    
    Args:
        vectors_all_layers: Vectors per layer [layer][sample][dim]
        save_path: Path to save file
    """
    # Convert to numpy arrays
    all_layers_hidden = [np.array(layer, dtype=np.float32) for layer in vectors_all_layers]
    
    # Save as compressed npz
    np.savez_compressed(
        save_path,
        **{f"layer_{idx}": layer for idx, layer in enumerate(all_layers_hidden)}
    )


def load_hidden_states(load_path: str) -> List[np.ndarray]:
    """Load hidden states from compressed numpy file.
    
    Args:
        load_path: Path to load file
        
    Returns:
        List of vectors per layer
    """
    data = np.load(load_path, allow_pickle=True)
    num_layers = len([k for k in data.keys() if k.startswith("layer_")])
    
    vectors_all_layers = []
    for idx in range(num_layers):
        vectors_all_layers.append(data[f"layer_{idx}"])
    
    return vectors_all_layers

