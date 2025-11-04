"""Analysis tools for PCA, hidden states, and visualization."""
from .pca import (
    pca_basic,
    get_regression_direction,
    get_safe_aligned_pc1,
    compute_anchor_pc1s,
    project_with_pc1,
    pca_projection_dot_mean
)
from .hidden_states import (
    extract_hidden_states,
    compute_cosine_similarity,
    extract_token_hidden_states,
    save_hidden_states,
    load_hidden_states
)
from .visualization import (
    pca_graph,
    plot_histogram,
    plot_layerwise_cosine
)

__all__ = [
    # PCA
    'pca_basic',
    'get_regression_direction',
    'get_safe_aligned_pc1',
    'compute_anchor_pc1s',
    'project_with_pc1',
    'pca_projection_dot_mean',
    # Hidden states
    'extract_hidden_states',
    'compute_cosine_similarity',
    'extract_token_hidden_states',
    'save_hidden_states',
    'load_hidden_states',
    # Visualization
    'pca_graph',
    'plot_histogram',
    'plot_layerwise_cosine',
]

