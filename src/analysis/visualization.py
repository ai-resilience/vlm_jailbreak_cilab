"""Visualization utilities for PCA and hidden states."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from typing import List, Optional, Dict
import os


def pca_graph(
    vectors: np.ndarray,
    labels: np.ndarray,
    pca_layer_index: int,
    save_path: str = "./result/pca.png"
) -> None:
    """Create PCA visualization.
    
    Args:
        vectors: Vectors to visualize [layer][sample][dim]
        labels: Labels for coloring
        pca_layer_index: Layer index to visualize, or "all" for grid
        save_path: Path to save figure
    """
    labels_ = np.array(labels)
    filename = save_path
    
    if isinstance(pca_layer_index, int):
        # Single layer PCA
        vecs = np.stack([v for v in vectors[pca_layer_index]])
        pca = PCA(n_components=2)
        vec_2d = pca.fit_transform(vecs)
        
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels_):
            idxs = np.where(labels_ == label)[0]
            plt.scatter(vec_2d[idxs, 0], vec_2d[idxs, 1], label=f"label={label}", alpha=0.6, s=5)
        
        plt.title(f"2D PCA of Layer {pca_layer_index}")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
    elif isinstance(pca_layer_index, str) and pca_layer_index.lower() == "all":
        # Grid of all layers
        vecs = np.array(vectors)
        num_layers = vecs.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_layers)))
        
        # Color by labels
        unique_labels = sorted(np.unique(labels_))
        cmap = cm.get_cmap("tab10", len(unique_labels))
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [color_map[l] for l in labels_]
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
        axes = axes.flatten()
        
        pca = PCA(n_components=2)
        for idx in range(num_layers):
            ax = axes[idx]
            vec_2d = pca.fit_transform(vecs[idx])
            ax.scatter(vec_2d[:, 0], vec_2d[:, 1], c=colors, alpha=0.7, s=5)
            ax.set_title(f"Layer {idx}")
            ax.axis("off")
        
        for idx in range(num_layers, grid_size * grid_size):
            fig.delaxes(axes[idx])
        
        # Legend
        handles = [
            Line2D([0], [0], marker='o', color='w', label=f"{label}",
                   markerfacecolor=color_map[label], markersize=6)
            for label in unique_labels
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, title="Label")
        plt.subplots_adjust(bottom=0.1)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    print(f"[✓] PCA plot saved to {filename}")


def plot_histogram(
    vectors: List[List[np.ndarray]],
    labels: np.ndarray,
    pca_layer_index: int,
    save_path: str = "./result/histogram.png",
    bins: int = 50,
    show_means: bool = True
) -> None:
    """Plot histogram of PC1 projections.
    
    Args:
        vectors: Vectors [layer][sample][dim]
        labels: Sample labels
        pca_layer_index: Layer to visualize or "all"
        save_path: Save path
        bins: Number of histogram bins
        show_means: Whether to show mean lines
    """
    try:
        from scipy.stats import gaussian_kde
        _HAS_KDE = True
    except Exception:
        _HAS_KDE = False
    
    def project_pc1(vecs_np: np.ndarray):
        pca = PCA(n_components=1)
        pc1 = pca.fit(vecs_np).components_[0]
        pc1 = pc1 / (np.linalg.norm(pc1) + 1e-9)
        proj = (vecs_np - pca.mean_) @ pc1
        return proj
    
    def plot_density(ax, data, color, lw=1.2, alpha=0.95):
        if _HAS_KDE and len(data) >= 2:
            try:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 400)
                ys = kde(xs)
                ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha)
                return float(np.max(ys))
            except Exception:
                pass
        hist, edges = np.histogram(data, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(centers, hist, color=color, linewidth=lw, alpha=alpha)
        return float(np.max(hist))
    
    label_colors = {0: "red", 1: "blue", 2: "orange", 3: "green"}
    labels_ = np.asarray(labels)
    
    if isinstance(pca_layer_index, int):
        vecs = np.stack([v for v in vectors[pca_layer_index]])
        proj = project_pc1(vecs)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for lab in np.unique(labels_):
            idxs = np.where(labels_ == lab)[0]
            plot_density(ax, proj[idxs], color=label_colors.get(lab, "gray"))
            if show_means and len(idxs) > 0:
                ax.axvline(proj[idxs].mean(), color=label_colors.get(lab, "gray"),
                          linestyle="--", linewidth=1.0, alpha=0.7)
        
        ax.set_title(f"PC1 Projection Distribution — Layer {pca_layer_index}")
        ax.set_xlabel("Projection onto PC1")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"[✓] Histogram saved to {save_path}")


def plot_layerwise_cosine(
    cos_list_0: List[List[float]],
    cos_list_1: List[List[float]],
    cos_list_2: Optional[List[List[float]]],
    model_name: str,
    save_path: str,
    mode: str = "harmful/harmless"
) -> None:
    """Plot layer-wise cosine similarity comparison.
    
    Args:
        cos_list_0: First cosine list per layer
        cos_list_1: Second cosine list per layer
        cos_list_2: Optional third cosine list
        model_name: Model name for title
        save_path: Save path
        mode: Comparison mode ("image/text" or "harmful/harmless")
    """
    L = len(cos_list_0)
    layer_xs = np.arange(L)
    mean_0 = [np.mean(cos_list_0[l]) for l in range(L)]
    mean_1 = [np.mean(cos_list_1[l]) for l in range(L)]
    
    plt.figure(figsize=(9, 4.5))
    
    if mode == "image/text":
        plt.plot(layer_xs, mean_0, color="red", linewidth=2, marker="o", label="Mean cos(img)", zorder=3)
        plt.plot(layer_xs, mean_1, color="blue", linewidth=2, marker="o", label="Mean cos(txt)", zorder=3)
        title = f"{model_name} — Cos(img) vs Cos(txt) per Layer"
    else:
        plt.plot(layer_xs, mean_0, color="red", linewidth=2, marker="o", label="Mean cos(harmful)", zorder=3)
        plt.plot(layer_xs, mean_1, color="blue", linewidth=2, marker="o", label="Mean cos(harmless)", zorder=3)
        if cos_list_2 is not None:
            mean_2 = [np.mean(cos_list_2[l]) for l in range(L)]
            plt.plot(layer_xs, mean_2, color="green", linewidth=2, marker="o", label="Mean cos(manual data)", zorder=3)
        title = f"{model_name} — Cos(harmful) vs Cos(harmless) per Layer"
    
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {save_path}")

