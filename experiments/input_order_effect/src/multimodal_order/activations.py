import numpy as np


def last_nonpadding_indices(attention_mask):
    """Return actual last attended positions for both left and right padding."""
    import torch

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = positions.masked_fill(attention_mask.to(dtype=torch.bool).logical_not(), -1)
    indices = masked.max(dim=1).values
    if (indices < 0).any():
        raise ValueError("every sample must contain at least one attended token")
    return indices


def extract_last_input_token(hidden_states, attention_mask) -> np.ndarray:
    """Stack each transformer layer's last real input-token representation."""
    import torch

    layers = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
    indices = last_nonpadding_indices(attention_mask)
    batch = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    values = torch.stack([layer[batch, indices, :] for layer in layers], dim=1)
    return torch.nan_to_num(values.float()).cpu().numpy()


def refusal_direction(alpaca_activations: np.ndarray) -> np.ndarray:
    """Paper-compatible harmless/refusal reference: mean Alpaca activation by layer."""
    if alpaca_activations.ndim != 3:
        raise ValueError("expected [samples, layers, hidden] Alpaca activations")
    return alpaca_activations.mean(axis=0)
