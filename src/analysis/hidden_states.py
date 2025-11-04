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
    inputs.to(model.device, dtype=torch.bfloat16)
    
    if model_name == "deepseek2":
        attention_mask = attention_mask.to(model.device)
        with torch.no_grad():
            out = model.language(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
    else:
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

