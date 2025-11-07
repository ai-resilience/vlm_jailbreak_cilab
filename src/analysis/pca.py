"""PCA analysis utilities."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List, Optional, Any


def pca_basic(vectors: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform basic PCA analysis.
    
    Args:
        vectors: Input vectors of shape (N, D)
        top_k: Number of principal components to return
        
    Returns:
        Tuple of (eigenvalues, eigenvectors, variance_ratio, mean)
        - eigenvalues: shape (top_k,) - variance
        - eigenvectors: shape (top_k, D) - principal components
        - variance_ratio: shape (top_k,) - explained variance ratio
        - mean: shape (D,) - data mean
    """
    pca = PCA(n_components=top_k)
    pca.fit(vectors)
    mean = pca.mean_
    return pca.explained_variance_, pca.components_, pca.explained_variance_ratio_, mean


def get_regression_direction(vectors: np.ndarray, labels: np.ndarray, top_k: int = 2) -> np.ndarray:
    """Get regression direction in high-dimensional space.
    
    Args:
        vectors: Input vectors of shape (N, D)
        labels: Binary labels of shape (N,)
        top_k: Number of PCs for regression
        
    Returns:
        Regression direction in original space (unit norm)
    """
    # PCA
    pca = PCA(n_components=top_k)
    pca_2d_proj = pca.fit_transform(vectors)
    eigen_vecs = pca.components_
    
    # Regression in PCA space
    clf = LogisticRegression()
    clf.fit(pca_2d_proj, labels)
    regression_2d_dir = clf.coef_[0]
    regression_2d_dir = regression_2d_dir / np.linalg.norm(regression_2d_dir)
    
    # Map back to original space
    regression_high_dim_dir = np.dot(regression_2d_dir, eigen_vecs)
    regression_high_dim_dir = regression_high_dim_dir / np.linalg.norm(regression_high_dim_dir)
    
    return regression_high_dim_dir


def get_safe_aligned_pc1(anchor_vecs: np.ndarray, anchor_labels: np.ndarray) -> np.ndarray:
    """Get PC1 aligned to safe direction.
    
    Args:
        anchor_vecs: Anchor vectors
        anchor_labels: Labels (0=unsafe, 1=safe)
        
    Returns:
        PC1 vector aligned so positive direction is safe
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit(anchor_vecs).components_[0]
    anchor_mean = pca.mean_
    
    unsafe_mean = anchor_vecs[np.asarray(anchor_labels) == 0].mean(axis=0)
    safe_mean = anchor_vecs[np.asarray(anchor_labels) == 1].mean(axis=0)
    
    a = (unsafe_mean - anchor_mean) @ pc1
    b = (safe_mean - anchor_mean) @ pc1
    
    if b > a:
        pc1 = -pc1
        
    return pc1


def compute_anchor_pc1s(anchor_all_layers: List[List[np.ndarray]], anchor_labels: np.ndarray) -> List[np.ndarray]:
    """Compute PC1 for each layer from anchor data.
    
    Args:
        anchor_all_layers: List of vectors per layer [layer][sample][dim]
        anchor_labels: Labels for anchor data
        
    Returns:
        List of PC1 vectors per layer
    """
    pc1_list = []
    for anchor_vectors in anchor_all_layers:
        anchor = np.stack(anchor_vectors)
        pc1 = get_safe_aligned_pc1(anchor, anchor_labels)
        pc1_list.append(pc1)
    return pc1_list


def project_with_pc1(pc1_list: List[np.ndarray], target_all_layers: List[List[np.ndarray]], layer_idx: int) -> List[List[float]]:
    """Project target vectors onto PC1 from specific layer.
    
    Args:
        pc1_list: List of PC1 vectors per layer
        target_all_layers: Target vectors [layer][sample][seq_len, dim]
        layer_idx: Layer index to use for PC1
        
    Returns:
        Projections for each layer
    """
    proj_all = [[] for _ in range(len(target_all_layers))]
    pc1 = pc1_list[layer_idx]
    
    for i in range(len(target_all_layers)):
        target_list = target_all_layers[i]
        for sample in target_list:
            tensor = torch.tensor(sample, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
            normed = tensor.detach().cpu().numpy()
            normed = normed / (np.linalg.norm(normed, axis=0, keepdims=True) + 1e-9)
            proj = (normed @ pc1).mean().item()
            proj_all[i].append(proj)
    
    return proj_all


def pca_projection_dot_mean(
    anchor_vectors: List[List[np.ndarray]], 
    target_vectors: List[List[np.ndarray]], 
    layer_indices: str = "all"
) -> List[Tuple[int, float]]:
    """Compute mean dot product of target projections on anchor PC1.
    
    Args:
        anchor_vectors: Anchor data [layer][sample][dim]
        target_vectors: Target data [layer][sample][dim]
        layer_indices: "all" or specific layer index
        
    Returns:
        List of (layer_index, mean_dot_product) tuples
    """
    def get_pc1(vecs):
        pca = PCA(n_components=1)
        pc1 = pca.fit(vecs).components_[0]
        return pc1 / (np.linalg.norm(pc1) + 1e-9)
    
    results = []
    
    if isinstance(layer_indices, int):
        anchor = np.stack([torch.tensor(v) for v in anchor_vectors[layer_indices]]).numpy()
        target = np.stack([torch.tensor(v) for v in target_vectors[layer_indices]]).numpy()
        pc1 = get_pc1(anchor)
        dot_mean = np.mean(target @ pc1)
        return dot_mean
    
    elif isinstance(layer_indices, str) and layer_indices.lower() == "all":
        num_layers = len(target_vectors)
        for i in range(num_layers):
            anchor = np.stack([torch.tensor(v) for v in anchor_vectors[i]]).numpy()
            target = np.stack([torch.tensor(v) for v in target_vectors[i]]).numpy()
            pc1 = get_pc1(anchor)
            dot_mean = np.mean(target @ pc1)
            results.append((i, dot_mean))
        return results
    
    raise ValueError("layer_indices must be int or 'all'")


def weight_cosine(vector: np.ndarray, model: Any, tokenizer: Any, top_k: int = 10) -> Tuple[List[str], torch.Tensor]:
    """Compute cosine similarity between PC1 vector and vocabulary embeddings.
    
    Args:
        vector: numpy array or torch tensor of shape (hidden_dim,)
        model: HuggingFace VLM model
        tokenizer: associated tokenizer
        top_k: number of most similar tokens to return
        
    Returns:
        Tuple of (tokens, scores)
        - tokens: List of top-k token strings
        - scores: Tensor of cosine similarity scores
    """
    from src.models.base import find_lm_head
    
    # Convert vector to tensor if needed
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)
    vector = vector.to(dtype=model.dtype, device=model.device)
    
    # Extract lm_head weights using unified function
    lm_head = find_lm_head(model)
    lm_weight = lm_head.weight.float()
    
    # Cosine similarity between vector and each vocab embedding
    sim = F.cosine_similarity(lm_weight, vector.unsqueeze(0), dim=1)  # [vocab_size]
    
    # Top-k most similar tokens
    topk = torch.topk(sim, k=top_k)
    indices = topk.indices
    scores = topk.values
    
    # Convert to list - indices.tolist() already returns Python ints, no need for .item()
    tokens = [tokenizer.decode([i]) for i in indices.tolist()]
    
    return tokens, scores

