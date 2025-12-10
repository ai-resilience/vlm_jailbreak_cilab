#!/usr/bin/env python3
"""Plot Figstep evaluation results: RR/ASR vs Size/Resolution/FontSize for different cases."""
import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Configuration
models = ["intern", "qwen", "llava_next", "kimi", "deepseek2", "phi"]  # Add more models as needed
datasets = ["mm_sd", "mm_sd_pad", "mm_typo_pad"]  # Three cases
metrics = ["ensemble", "llamaguard4", "beaverdam", "wildguard"]  # Metrics to plot

# Base directory (relative to script location)
# Script is in: vlm_jailbreak_cilab/scripts/analysis/
# Result is in: vlm_kihyun_workspace/result/
BASE_DIR = "../../../result/evaluation"
OUTPUT_DIR = "../../../result/plots"


def extract_value_from_filename(filename: str, dataset: str) -> int:
    """Extract size/resolution/font_size from filename based on dataset type.
    
    Args:
        filename: Filename to parse
        dataset: Dataset name ('Figstep', 'Figstep_font', 'Figstep_res', 'mm_sd', 'mm_sd_pad', or 'mm_typo_pad')
        
    Returns:
        Extracted value (size/resolution/font_size) or None
    """
    if dataset == "mm_sd":
        # Pattern: {model}_image_mm_sd_response_{resolution}_{metric}_result.txt
        # Support both: response_{resolution}_{metric} and response_{resolution}_ensemble
        match = re.search(r"response_(\d+)_(?:llamaguard4|beaverdam|wildguard|ensemble)", filename)
        if match:
            return int(match.group(1))
    
    elif dataset == "mm_sd_pad":
        # Pattern: {model}_image_mm_sd_response_padded_{resolution}_{metric}_result.txt
        match = re.search(r"response_padded_(\d+)_(?:llamaguard4|beaverdam|wildguard|ensemble)", filename)
        if match:
            return int(match.group(1))
    
    elif dataset == "mm_typo_pad":
        # Pattern: {model}_image_mm_typo_response_padded_{resolution}_{metric}_result.txt
        match = re.search(r"response_padded_(\d+)_(?:llamaguard4|beaverdam|wildguard|ensemble)", filename)
        if match:
            return int(match.group(1))
    
    return None


def extract_rate_from_file(filepath: str, metric: str) -> float:
    """Extract rate (RR or ASR) from evaluation result file.
    
    Args:
        filepath: Path to result.txt file
        metric: 'keyword', 'llamaguard4', 'beaverdam', 'wildguard', or 'ensemble'
        
    Returns:
        Rate as float (percentage)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if metric == 'keyword':
            # Pattern: 🔥 RR (Refusal Rate) (Rate): 21.20%
            match = re.search(r"RR.*?\(Refusal Rate\).*?([\d.]+)%", text)
        else:  # llamaguard4, beaverdam, wildguard, ensemble
            # Pattern: 🔥 Unsafe Rate (Rate): 1.60% or ASR (Attack Success Rate): 1.60%
            match = re.search(r"(?:ASR|Unsafe Rate).*?([\d.]+)%", text)
        
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
    
    return None


def get_xlabel_and_title_suffix(dataset: str) -> tuple:
    """Get xlabel and title suffix based on dataset type.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Tuple of (xlabel, title_suffix)
    """
    if dataset == "Figstep":
        return "Resolution", "Resolution"
    elif dataset == "Figstep_font":
        return "Font Size", "Font Size"
    elif dataset == "Figstep_res":
        return "Size", "Size"
    elif dataset in ["mm_sd", "mm_sd_pad", "mm_typo_pad"]:
        return "Resolution", "Resolution"
    return "Value", "Value"


def plot_value_vs_rate(
    model: str,
    dataset: str,
    metric: str,
    results: dict,
    show_plot: bool = False
):
    """Plot value (size/resolution/font_size) vs rate (RR/ASR) for a model/metric/dataset combination.
    
    Args:
        model: Model name
        dataset: Dataset name
        metric: Metric name ('keyword', 'llamaguard4', 'beaverdam', 'wildguard', or 'ensemble')
        results: Dict mapping value -> rate
        show_plot: Whether to display the plot
    """
    if not results:
        print(f"⚠️  No data found for {model}/{dataset}/{metric}")
        return
    
    # Determine plot title and ylabel
    if metric == 'keyword':
        ylabel = "RR (Refusal Rate) (%)"
        title_suffix = "Refusal Rate"
    else:  # llamaguard4, beaverdam
        ylabel = "ASR (Attack Success Rate) (%)"
        title_suffix = "Attack Success Rate"
    
    xlabel, xlabel_suffix = get_xlabel_and_title_suffix(dataset)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Sort by value
    sorted_items = sorted(results.items())
    values, rates = zip(*sorted_items) if sorted_items else ([], [])
    
    if not values:
        print(f"⚠️  No valid data points for {model}/{dataset}/{metric}")
        return
    
    x_pos = range(len(values))
    plt.plot(x_pos, rates, marker='o', label=model.capitalize(), linewidth=2, markersize=8, color='steelblue')
    plt.xticks(x_pos, values)
    
    plt.title(f"{title_suffix} vs {xlabel_suffix} ({model}, {dataset}, {metric})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save plot in dataset-specific folder
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    save_path = os.path.join(dataset_output_dir, f"{model}_{dataset}_{metric}_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_models_comparison(
    dataset: str,
    metric: str,
    results_by_model: dict,
    show_plot: bool = False
):
    """Plot value vs rate for all models in one plot for comparison.
    
    Args:
        dataset: Dataset name
        metric: Metric name ('keyword', 'llamaguard4', 'beaverdam', 'wildguard', or 'ensemble')
        results_by_model: Dict mapping model -> {value: rate}
        show_plot: Whether to display the plot
    """
    if not results_by_model:
        print(f"⚠️  No data found for {dataset}/{metric}")
        return
    
    # Determine plot title and ylabel
    if metric == 'keyword':
        ylabel = "RR (Refusal Rate) (%)"
        title_suffix = "Refusal Rate"
    else:  # llamaguard4, beaverdam
        ylabel = "ASR (Attack Success Rate) (%)"
        title_suffix = "Attack Success Rate"
    
    xlabel, xlabel_suffix = get_xlabel_and_title_suffix(dataset)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Get all unique values across all models
    all_values = set()
    for results in results_by_model.values():
        all_values.update(results.keys())
    all_values = sorted(all_values)
    
    if not all_values:
        print(f"⚠️  No valid data points for {dataset}/{metric}")
        return
    
    # Colors for different models
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange', 'red', 'brown', 'pink']
    
    x_pos = range(len(all_values))
    for idx, (model, results) in enumerate(results_by_model.items()):
        if not results:
            continue
        
        # Get rates for each value (None if not available)
        rates = [results.get(v, None) for v in all_values]
        
        # Filter out None values for plotting
        valid_indices = [i for i, r in enumerate(rates) if r is not None]
        valid_x = [x_pos[i] for i in valid_indices]
        valid_rates = [rates[i] for i in valid_indices]
        
        if valid_rates:
            color = colors[idx % len(colors)]
            plt.plot(valid_x, valid_rates, marker='o', label=model.capitalize(), 
                    linewidth=2, markersize=8, color=color)
    
    plt.xticks(x_pos, all_values)
    plt.title(f"{title_suffix} vs {xlabel_suffix} ({dataset}, {metric}) - Model Comparison")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save plot in dataset-specific folder
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    save_path = os.path.join(dataset_output_dir, f"{dataset}_{metric}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def process_dataset(dataset: str, metric: str):
    """Process a single dataset and metric combination.
    
    Args:
        dataset: Dataset name
        metric: Metric name
    """
    print(f"\n📊 Processing: {dataset} / {metric}")
    
    # Results organized by model
    results_by_model = {model: {} for model in models}
    
    # Process each model
    for model in models:
        base_dir = os.path.join(BASE_DIR, model, dataset)
        
        if not os.path.exists(base_dir):
            print(f"  ⚠️  Directory not found: {base_dir}")
            continue
        
        print(f"  📁 Processing: {model}/{dataset}")
        
        # Build pattern based on dataset type
        if dataset == "Figstep":
            # Pattern: {model}_image_Figstep_response_{resolution}_{metric}_result.txt
            # For ensemble: {model}_image_Figstep_response_{resolution}_ensemble_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_image_Figstep_response_\d+_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_image_Figstep_response_\d+_{re.escape(metric)}_result\.txt$"
                )
        elif dataset == "Figstep_font":
            # Pattern: {model}_image_Figstep_font_font{size}_response_{metric}_result.txt
            # For ensemble: {model}_image_Figstep_font_font{size}_response_ensemble_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_image_Figstep_font_font\d+_response_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_image_Figstep_font_font\d+_response_{re.escape(metric)}_result\.txt$"
                )
        elif dataset == "Figstep_res":
            # Pattern: {model}_Figstep_size{number}_response_{metric}_result.txt
            # For ensemble: {model}_Figstep_size{number}_response_ensemble_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_Figstep_size\d+_response_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_Figstep_size\d+_response_{re.escape(metric)}_result\.txt$"
                )
        elif dataset == "mm_sd":
            # Pattern: {model}_image_mm_sd_response_{resolution}_{metric}_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_sd_response_\d+_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_sd_response_\d+_{re.escape(metric)}_result\.txt$"
                )
        elif dataset == "mm_sd_pad":
            # Pattern: {model}_image_mm_sd_response_padded_{resolution}_{metric}_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_sd_response_padded_\d+_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_sd_response_padded_\d+_{re.escape(metric)}_result\.txt$"
                )
        elif dataset == "mm_typo_pad":
            # Pattern: {model}_image_mm_typo_response_padded_{resolution}_{metric}_result.txt
            if metric == "ensemble":
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_typo_response_padded_\d+_ensemble_result\.txt$"
                )
            else:
                pattern = re.compile(
                    rf"{re.escape(model)}_image_mm_typo_response_padded_\d+_{re.escape(metric)}_result\.txt$"
                )
        else:
            print(f"  ⚠️  Unknown dataset: {dataset}")
            continue
        
        matched_files = []
        for fname in os.listdir(base_dir):
            if not fname.endswith(f"_{metric}_result.txt"):
                continue
            
            if not pattern.match(fname):
                continue
            
            matched_files.append(fname)
            
            # Extract value (size/resolution/font_size)
            value = extract_value_from_filename(fname, dataset)
            if value is None:
                print(f"    ⚠️  Could not extract value from: {fname}")
                continue
            
            # Extract rate
            filepath = os.path.join(base_dir, fname)
            rate = extract_rate_from_file(filepath, metric)
            
            if rate is None:
                print(f"    ⚠️  Could not extract rate from: {fname}")
                continue
            
            # Store result
            results_by_model[model][value] = rate
            xlabel, _ = get_xlabel_and_title_suffix(dataset)
            print(f"    ✓ {xlabel} {value}: {rate:.2f}%")
        
        if not matched_files:
            print(f"    ⚠️  No files matched pattern for {metric}")
        else:
            print(f"    📊 Found {len(matched_files)} files for {metric}")
        
        # Generate individual plot for this model
        if results_by_model[model]:
            plot_value_vs_rate(
                model=model,
                dataset=dataset,
                metric=metric,
                results=results_by_model[model],
                show_plot=False
            )
    
    # Generate comparison plot for all models
    if any(results_by_model.values()):
        plot_all_models_comparison(
            dataset=dataset,
            metric=metric,
            results_by_model=results_by_model,
            show_plot=False
        )
    else:
        print(f"  ⚠️  No data found for {dataset}/{metric}")


def main():
    """Main function to process all datasets/metrics and generate plots."""
    print("🚀 Starting Figstep evaluation result plotting...")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"📊 Output directory: {OUTPUT_DIR}\n")
    
    # Process each dataset and metric combination
    for dataset in datasets:
        for metric in metrics:
            process_dataset(dataset, metric)
    
    print("\n✅ All plots generated!")


if __name__ == '__main__':
    main()
