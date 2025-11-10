#!/usr/bin/env python3
"""Plot evaluation results: ASR/RR vs Image Resolution."""
import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Configuration
models = ["kimi", "llava_next", "intern", "qwen", "deepseek2"]
datasets = ["mm_sd_typo", "Figstep"]
image_types = ["blank"]  # Optional: filter by image type
metrics = ["keyword"]  # Metrics to plot /  "llamaguard4"

# Base directory (relative to project root)
BASE_DIR = "../result/evaluation"
OUTPUT_DIR = "../result/plots"


def extract_resolution_from_filename(filename: str) -> int:
    """Extract resolution from filename.
    
    Pattern: {model}_{modality}_{dataset}_response_{resolution}_{metric}_result.txt
    Example: deepseek2_image_Figstep_response_768_keyword_result.txt
    """
    match = re.search(r"response_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def extract_image_type_from_filename(filename: str) -> str:
    """Extract image type from filename if present.
    
    Pattern: {model}_{modality}_{dataset}_response_{image_type}_{resolution}_{metric}_result.txt
    Example: qwen_image_StrongREJECT_response_blank_1120_keyword_result.txt
    """
    # Check for image types in filename
    for img_type in image_types:
        if f"_{img_type}_" in filename:
            return img_type
    return None


def extract_rate_from_file(filepath: str, metric: str) -> float:
    """Extract rate (RR or ASR) from evaluation result file.
    
    Args:
        filepath: Path to result.txt file
        metric: 'keyword' or 'llamaguard4'
        
    Returns:
        Rate as float (percentage)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if metric == 'keyword':
            # Pattern: 🔥 RR (Refusal Rate) (Rate): 1.60%
            match = re.search(r"RR.*?\(Refusal Rate\).*?([\d.]+)%", text)
        else:  # llamaguard4
            # Pattern: 🔥 Unsafe Rate (Rate): 1.60% or ASR (Attack Success Rate): 1.60%
            match = re.search(r"(?:ASR|Unsafe Rate).*?([\d.]+)%", text)
        
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
    
    return None


def plot_resolution_vs_rate(
    model: str,
    dataset: str,
    metric: str,
    results_by_image_type: dict = None,
    show_plot: bool = False
):
    """Plot resolution vs rate (RR/ASR) for a model/dataset/metric combination.
    
    Args:
        model: Model name
        dataset: Dataset name
        metric: Metric name ('keyword' or 'llamaguard4')
        results_by_image_type: Dict mapping image_type -> {resolution: rate}
        show_plot: Whether to display the plot
    """
    if not results_by_image_type:
        print(f"⚠️  No data found for {model}/{dataset}/{metric}")
        return
    
    # Determine plot title and ylabel
    if metric == 'keyword':
        ylabel = "RR (Refusal Rate) (%)"
        title_suffix = "Refusal Rate"
    else:  # llamaguard4
        ylabel = "ASR (Attack Success Rate) (%)"
        title_suffix = "Attack Success Rate"
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for img_type, results in results_by_image_type.items():
        if not results:
            continue
        
        # Sort by resolution
        sorted_items = sorted(results.items())
        resolutions, rates = zip(*sorted_items) if sorted_items else ([], [])
        
        if not resolutions:
            continue
        
        x_pos = range(len(resolutions))
        label = f"{img_type.capitalize()}" if img_type else "All"
        plt.plot(x_pos, rates, marker='o', label=label, linewidth=2, markersize=8)
        plt.xticks(x_pos, resolutions)
    
    plt.title(f"{title_suffix} vs Image Resolution ({model}, {dataset})")
    plt.xlabel("Image Resolution")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{model}_{dataset}_{metric}_resolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to process all models/datasets and generate plots."""
    print("🚀 Starting evaluation result plotting...")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"📊 Output directory: {OUTPUT_DIR}\n")
    
    for model in models:
        for dataset in datasets:
            base_dir = os.path.join(BASE_DIR, model, dataset)
            
            if not os.path.exists(base_dir):
                print(f"⚠️  Directory not found: {base_dir}")
                continue
            
            print(f"📁 Processing: {model}/{dataset}")
            
            # Process each metric
            for metric in metrics:
                # Results organized by image type
                results_by_image_type = {img_type: {} for img_type in image_types}
                results_by_image_type[None] = {}  # For files without image type
                
                # Find all result files for this metric
                pattern = re.compile(
                    rf"{re.escape(model)}_.*_{re.escape(dataset)}_response_.*_{re.escape(metric)}_result\.txt$"
                )
                
                for fname in os.listdir(base_dir):
                    if not pattern.match(fname):
                        continue
                    
                    # Extract resolution
                    resolution = extract_resolution_from_filename(fname)
                    if resolution is None:
                        continue
                    
                    # Extract image type (if present)
                    img_type = extract_image_type_from_filename(fname)
                    
                    # Extract rate
                    filepath = os.path.join(base_dir, fname)
                    rate = extract_rate_from_file(filepath, metric)
                    
                    if rate is None:
                        continue
                    
                    # Store result
                    if img_type:
                        results_by_image_type[img_type][resolution] = rate
                    results_by_image_type[None][resolution] = rate
                
                # Generate plot
                if any(results_by_image_type.values()):
                    plot_resolution_vs_rate(
                        model=model,
                        dataset=dataset,
                        metric=metric,
                        results_by_image_type=results_by_image_type,
                        show_plot=False
                    )
                else:
                    print(f"  ⚠️  No data found for {metric}")
    
    print("\n✅ All plots generated!")


if __name__ == '__main__':
    main()
