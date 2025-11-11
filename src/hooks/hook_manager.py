"""Hook manager for injecting activations during model forward pass."""
import torch
from typing import List, Dict, Any
from ..models.base import find_layers


class HookManager:
    """Manages forward hooks for activation injection."""
    
    def __init__(
        self,
        model: Any,
        all_layer_eigen_vecs: List[torch.Tensor],
        layer_indices: int,
        token_indices: int,
        alpha: float,
        max_uses: int = 1
    ):
        """Initialize hook manager.
        
        Args:
            model: The model to attach hooks to
            all_layer_eigen_vecs: PC1 vectors for each layer
            layer_indices: Layer index to attach hook
            token_indices: Token position to modify
            alpha: Scaling factor for injection
            max_uses: Number of forward passes before auto-removal
        """
        self.fns = []
        self.handles = []
        self.states = []
        
        # Attach hook to specified layer
        pc1_vector = torch.tensor(all_layer_eigen_vecs[layer_indices]).float()
        hook_fn, state = self._create_injection_hook(
            pc1_vector, alpha, max_uses, token_indices
        )
        
        # Find layers module (handles both language_model.layers and language_model.model.layers)
        layers = find_layers(model)
        
        # Register hook
        h = layers[layer_indices].register_forward_hook(hook_fn)
        state['handle'] = h
        
        self.handles.append(h)
        self.fns.append(hook_fn)
        self.states.append(state)
    
    def _create_injection_hook(
        self,
        pc1_vector: torch.Tensor,
        alpha: float,
        max_uses: int,
        token_indices: int
    ) -> tuple:
        """Create a forward hook for PC1 injection.
        
        Args:
            pc1_vector: PC1 direction vector
            alpha: Scaling factor
            max_uses: Max number of uses
            token_indices: Token position to modify
            
        Returns:
            Tuple of (hook_fn, state_dict)
        """
        state = {
            'uses': 0,
            'max_uses': int(max_uses),
            'removed': False,
            'handle': None,
            'pc1': pc1_vector.clone()
        }
        
        def hook_fn(module, input, output):
            if state['removed']:
                return output
            
            # Get hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            pc1 = state['pc1'].to(hidden.device).to(hidden.dtype)
            
            # Check if max uses reached
            if state['uses'] >= state['max_uses']:
                h = state.get('handle', None)
                if h is not None:
                    try:
                        h.remove()
                    except Exception:
                        pass
                state['removed'] = True
                return output
            
            # Inject PC1 at all token positions
            pc1_row = pc1.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            norms = hidden.norm(dim=-1, keepdim=True)  # (B, T, 1)
            scaled_alpha = alpha * norms
            add_term = scaled_alpha * pc1_row.to(hidden.device)
            hidden[:, :, :] = hidden[:, :, :] + add_term[:, :, :]
            
            # Increment usage
            state['uses'] += 1
            
            # Auto-remove if limit reached
            if state['uses'] >= state['max_uses']:
                h = state.get('handle', None)
                if h is not None:
                    try:
                        h.remove()
                    except Exception:
                        pass
                state['removed'] = True
            
            return output
        
        return hook_fn, state
    
    def reset(self):
        """Reset usage counters."""
        for state in self.states:
            state['uses'] = 0
            state['removed'] = False
    
    def remove(self):
        """Remove all hooks."""
        for h in list(self.handles):
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        
        for state in self.states:
            state['removed'] = True

