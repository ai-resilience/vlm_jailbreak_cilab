#!/usr/bin/env python3
"""PCA analysis on hidden states from multiple data classes."""
import sys
import os
import json
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models import load_model
from src.analysis.hidden_states import extract_hidden_states, extract_token_hidden_states
from src.inference.processor import build_prompt
from src.models.base import find_norm, find_num_hidden_layers


# Data paths
LAION_COCO_SAFE_PATH = "/mnt/server8_hard3/wonjun/laion-coco_safe_unsafe/safe_0.05"
LAION_COCO_UNSAFE_PATH = "/mnt/server8_hard3/wonjun/laion-coco_safe_unsafe/unsafe_0.95"
SAFEGUARD_JSONL_PATH = "/mnt/server16_hard1/kihyun/vlm_kihyun_workspace/vlm_jailbreak_cilab/dataset/llmsafeguard_dataset.jsonl"
VLGUARD_IMAGE_DIR = "/mnt/server4_hard0/kihyun/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models-Ours/kihyun/dataset/VLGuard/hatefulMemes"
FIGSTEP_UNSAFE_IMAGE_DIR = "/mnt/server16_hard1/wonjun/lvlm_jailbreak/dataset/FigStep/data/images/no_numbering/SafeBench"
FIGSTEP_SAFE_IMAGE_DIR = "/mnt/server16_hard1/wonjun/lvlm_jailbreak/dataset/FigStep/data/images/no_numbering/SafeBench_contradict"
FIGSTEP_UNSAFE_TEXT_PATH = "/mnt/server16_hard1/wonjun/lvlm_jailbreak/dataset/FigStep/data/question/safebench.csv"
FIGSTEP_SAFE_TEXT_PATH = "/mnt/server16_hard1/wonjun/lvlm_jailbreak/dataset/FigStep/data/question/safebench_contradict.csv"

def parse_class_string(class_str: str) -> Tuple[str, str, str]:
    """Parse class string 'modality:type:safety' into tuple.
    
    Args:
        class_str: Class string in format 'modality:type:safety'
        
    Returns:
        Tuple of (modality, type, safety)
    """
    parts = class_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid class format: {class_str}. Expected 'modality:type:safety'")
    return tuple(parts)


def load_laion_coco_images(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load LAION-COCO images from directory.
    
    Args:
        data_dir: Directory containing images
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dicts with 'image_path' and 'prompt' keys
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Directory not found: {data_dir}")
        return []
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    # For images, we need a prompt - use a generic prompt
    prompt = ""
    
    samples = []
    for img_path in image_files:
        samples.append({
            'image_path': img_path,
            'prompt': prompt,
            'modality': 'image',
            'type': 'laion_coco',
            'safety': 'safe' if 'safe' in data_dir else 'unsafe'
        })
    
    return samples


def load_safeguard_texts(jsonl_path: str, label_filter: Optional[int] = None, max_samples: Optional[int] = None) -> List[Dict]:
    """Load LLMSafeguard texts from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        label_filter: Filter by label (0=unsafe, 1=safe, None=all)
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dicts with 'prompt' and 'label' keys
    """
    if not os.path.exists(jsonl_path):
        print(f"Warning: File not found: {jsonl_path}")
        return []
    
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            
            data = json.loads(line.strip())
            label = data.get('label', 0)
            
            if label_filter is not None and label != label_filter:
                continue
            
            samples.append({
                'prompt': data.get('prompt', ''),
                'image_path': None,
                'modality': 'text',
                'type': 'safeguard',
                'safety': 'safe' if label == 1 else 'unsafe'
            })
    
    return samples


def load_figstep_images(image_dir: str, safety: str = 'unsafe', max_samples: Optional[int] = None) -> List[Dict]:
    """Load FigStep images from directory (images only, no prompts).
    
    Args:
        image_dir: Directory containing images
        safety: 'safe' or 'unsafe'
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dicts with 'image_path', 'prompt', 'modality', 'type', 'safety'
    """
    if not os.path.exists(image_dir):
        print(f"Warning: Directory not found: {image_dir}")
        return []
    
    # Load image files
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    samples = []
    for img_path in image_files:
        samples.append({
            'image_path': img_path,
            'prompt': "",  # Empty prompt for images only
            'modality': 'image',
            'type': 'figstep',
            'safety': safety
        })
    
    return samples


def load_figstep_texts(csv_path: str, safety: str = 'unsafe', max_samples: Optional[int] = None) -> List[Dict]:
    """Load FigStep texts from CSV file (texts only, no images).
    
    Args:
        csv_path: Path to CSV file with question/prompt data
        safety: 'safe' or 'unsafe'
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dicts with 'prompt', 'image_path', 'modality', 'type', 'safety'
    """
    import csv
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return []
    
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_samples and len(samples) >= max_samples:
                break
            
            question = row.get('question', '')
            if question:
                samples.append({
                    'prompt': question,
                    'image_path': None,  # No image for texts only
                    'modality': 'text',
                    'type': 'figstep',
                    'safety': safety
                })
    
    return samples


def load_vlguard_images(image_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load VLGuard images from directory.
    
    Args:
        image_dir: Directory containing images
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dicts with 'image_path' and 'prompt' keys
    """
    if not os.path.exists(image_dir):
        print(f"Warning: Directory not found: {image_dir}")
        return []
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    # For VLGuard, use a generic prompt
    prompt = ""
    
    samples = []
    for img_path in image_files:
        samples.append({
            'image_path': img_path,
            'prompt': prompt,
            'modality': 'image',
            'type': 'VLGuard',
            'safety': 'unsafe'
        })
    
    return samples


def load_data_classes(selected_classes: Optional[List[Tuple[str, str, str]]] = None, max_samples_per_class: Optional[int] = None) -> List[Tuple[str, str, str, List[Dict]]]:
    """Load selected data classes with metadata.
    
    Args:
        selected_classes: List of (modality, type, safety) tuples to load.
                         If None, load all classes.
        max_samples_per_class: Maximum samples per class (None for all)
        
    Returns:
        List of (modality, type, safety, samples) tuples
    """
    all_classes = [
        ('image', 'laion_coco', 'safe'),
        ('image', 'laion_coco', 'unsafe'),
        ('text', 'safeguard', 'safe'),
        ('text', 'safeguard', 'unsafe'),
        ('image', 'figstep', 'safe'),
        ('image', 'figstep', 'unsafe'),
        ('text', 'figstep', 'safe'),
        ('text', 'figstep', 'unsafe'),
        ('image', 'VLGuard', 'unsafe'),
    ]
    
    if selected_classes is None:
        selected_classes = all_classes
    else:
        # Validate selected classes
        for cls in selected_classes:
            if cls not in all_classes:
                print(f"Warning: Unknown class {cls}, skipping")
        selected_classes = [cls for cls in selected_classes if cls in all_classes]
    
    results = []
    
    for modality, type_name, safety in selected_classes:
        samples = []
        
        if (modality, type_name, safety) == ('image', 'laion_coco', 'safe'):
            samples = load_laion_coco_images(LAION_COCO_SAFE_PATH, max_samples_per_class)
        elif (modality, type_name, safety) == ('image', 'laion_coco', 'unsafe'):
            samples = load_laion_coco_images(LAION_COCO_UNSAFE_PATH, max_samples_per_class)
        elif (modality, type_name, safety) == ('text', 'safeguard', 'safe'):
            samples = load_safeguard_texts(SAFEGUARD_JSONL_PATH, label_filter=1, max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('text', 'safeguard', 'unsafe'):
            samples = load_safeguard_texts(SAFEGUARD_JSONL_PATH, label_filter=0, max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('image', 'figstep', 'safe'):
            samples = load_figstep_images(FIGSTEP_SAFE_IMAGE_DIR, safety='safe', max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('image', 'figstep', 'unsafe'):
            samples = load_figstep_images(FIGSTEP_UNSAFE_IMAGE_DIR, safety='unsafe', max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('text', 'figstep', 'safe'):
            samples = load_figstep_texts(FIGSTEP_SAFE_TEXT_PATH, safety='safe', max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('text', 'figstep', 'unsafe'):
            samples = load_figstep_texts(FIGSTEP_UNSAFE_TEXT_PATH, safety='unsafe', max_samples=max_samples_per_class)
        elif (modality, type_name, safety) == ('image', 'VLGuard', 'unsafe'):
            samples = load_vlguard_images(VLGUARD_IMAGE_DIR, max_samples_per_class)
        
        if samples:
            results.append((modality, type_name, safety, samples))
            print(f"Loaded {len(samples)} samples for {modality}:{type_name}:{safety}")
        else:
            print(f"No samples loaded for {modality}:{type_name}:{safety}")
    
    return results


def get_instruction_token(model_name: str) -> Optional[str]:
    """Get instruction token for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Instruction token string or None
    """
    inst_tokens = {
        'intern': '<|im_end|>\n<|im_start|>assistant\n',
        'llava_next': '[/INST]',
        'qwen': '<|im_end|>\n<|im_start|>assistant',
    }
    return inst_tokens.get(model_name)


def get_pre_instruction_position(model_name: str, tokenizer: Any, input_ids: torch.Tensor) -> Optional[int]:
    """Find token position right before instruction tokens.
    
    Args:
        model_name: Name of the model
        tokenizer: Tokenizer instance
        input_ids: Tokenized input IDs [batch, seq_len] or [seq_len]
        
    Returns:
        Token position index (relative position, negative index) or None if not found
    """
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    
    inst_token = get_instruction_token(model_name)
    if not inst_token:
        return None
    
    # Tokenize instruction token
    try:
        tokenized_inst = tokenizer(inst_token, return_tensors='pt', add_special_tokens=False)
        inst_token_ids = tokenized_inst.input_ids[0]
        inst_token_ids = inst_token_ids.to(input_ids.device)
        inst_token_len = len(inst_token_ids)
    except:
        return None
    
    # Find instruction token in input_ids
    seq_len = len(input_ids)
    for i in range(seq_len - inst_token_len + 1):
        if torch.equal(input_ids[i:i+inst_token_len], inst_token_ids):
            # Pre-instruction position is right before the instruction token
            return i - 1
    
    return None


def get_input_ids_for_intern(model: Any, processor: Any, prompt: str, image_path: Optional[str]) -> Optional[torch.Tensor]:
    """Get input_ids for intern model by replicating extract_hidden_states logic.
    
    Args:
        model: The VLM model
        processor: Model processor
        prompt: Text prompt
        image_path: Path to image (or None)
        
    Returns:
        input_ids tensor or None
    """
    try:
        from src.models.intern.preprocess import load_image
        
        prompt_text = prompt
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
        else:
            pixel_values = None
        
        # Prepare template and query
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
        
        return input_ids
    except Exception as e:
        print(f"Error getting input_ids for intern: {e}")
        return None


def extract_hidden_states_at_positions(
    model: Any,
    processor: Any,
    model_name: str,
    prompt: str,
    image_path: Optional[str],
    norm_layer: Any,
    num_layers: int
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Extract hidden states at pre-instruction and last token positions.
    
    Args:
        model: The VLM model
        processor: Model processor
        model_name: Name of the model
        prompt: Text prompt
        image_path: Path to image (or None)
        norm_layer: Normalization layer
        num_layers: Number of layers
        
    Returns:
        Tuple of (pre_inst_vectors, last_token_vectors)
        Each is a list of vectors per layer, or None if extraction failed
    """
    try:
        # Extract hidden states first
        hidden_states = extract_hidden_states(model, processor, model_name, prompt, image_path, build_prompt)
        
        # Extract last token position
        last_token_vectors = extract_token_hidden_states(hidden_states, norm_layer, token_index=-1, num_layers=num_layers)
        
        # Get input_ids for finding instruction position
        input_ids = None
        if model_name == "intern":
            input_ids = get_input_ids_for_intern(model, processor, prompt, image_path)
        else:
            # Build prompt to get input_ids
            try:
                inputs, attention_mask = build_prompt(model, processor, model_name, image_path, prompt)
                
                if model_name == "deepseek2":
                    input_ids = attention_mask.input_ids if hasattr(attention_mask, 'input_ids') else None
                elif isinstance(inputs, dict):
                    input_ids = inputs.get('input_ids', None)
                elif hasattr(inputs, 'input_ids'):
                    input_ids = inputs.input_ids
            except Exception:
                # If building prompt fails (e.g., image processing error), skip pre-instruction extraction
                input_ids = None
        
        # Extract pre-instruction position
        pre_inst_vectors = None
        if input_ids is not None:
            try:
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                pre_inst_pos = get_pre_instruction_position(model_name, tokenizer, input_ids)
                
                if pre_inst_pos is not None:
                    pre_inst_vectors = extract_token_hidden_states(hidden_states, norm_layer, token_index=pre_inst_pos, num_layers=num_layers)
            except Exception:
                # Skip pre-instruction extraction if it fails
                pass
        
        return pre_inst_vectors, last_token_vectors
    
    except Exception:
        # Silently skip samples with errors
        return None, None


def create_pca_plot_all_layers(
    vectors_all_layers: List[List[np.ndarray]],
    class_labels: List[int],
    class_names: List[str],
    position_name: str,
    save_path: str
) -> None:
    """Create PCA plot for all layers in a grid.
    
    Args:
        vectors_all_layers: Vectors per layer [layer][sample][dim]
        class_labels: Class label for each sample
        class_names: Class name for each sample (for legend)
        position_name: Position name ('pre_instruction' or 'last_token')
        save_path: Path to save figure
    """
    num_layers = len(vectors_all_layers)
    if num_layers == 0:
        print("Warning: No layers to plot")
        return
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_layers)))
    
    # Get unique classes
    unique_classes = sorted(set(class_labels))
    num_classes = len(unique_classes)
    
    # Define colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_classes)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Create color and marker maps
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_classes)}
    marker_map = {label: markers[i % len(markers)] for i, label in enumerate(unique_classes)}
    
    # Create figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each layer
    pca = PCA(n_components=2)
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if not vectors_all_layers[layer_idx]:
            ax.axis('off')
            continue
        
        vecs = np.stack(vectors_all_layers[layer_idx])  # [num_samples, dim]
        vec_2d = pca.fit_transform(vecs)
        
        # Plot each class
        for class_label in unique_classes:
            idxs = np.where(np.array(class_labels) == class_label)[0]
            if len(idxs) == 0:
                continue
            
            # Get class name (use first occurrence)
            class_name = class_names[idxs[0]] if idxs[0] < len(class_names) else f"Class {class_label}"
            
            ax.scatter(
                vec_2d[idxs, 0], vec_2d[idxs, 1],
                label=class_name if layer_idx == 0 else "",  # Only label in first subplot
                alpha=0.7,
                s=30,
                c=[color_map[class_label]],
                marker=marker_map[class_label],
                edgecolors='black',
                linewidths=0.3
            )
        
        ax.set_title(f"Layer {layer_idx}", fontsize=10, fontweight='bold')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(num_layers, grid_size * grid_size):
        fig.delaxes(axes[idx])
    
    # Add legend
    handles = []
    for label in unique_classes:
        # Find first occurrence of this label
        label_idx = None
        for i, lbl in enumerate(class_labels):
            if lbl == label:
                label_idx = i
                break
        
        class_name = class_names[label_idx] if label_idx is not None and label_idx < len(class_names) else f"Class {label}"
        
        handles.append(
            plt.Line2D([0], [0], marker=marker_map[label], color='w',
                      label=class_name,
                      markerfacecolor=color_map[label], markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5)
        )
    fig.legend(handles=handles, loc='lower center', ncol=min(5, len(unique_classes)), 
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f"2D PCA - All Layers ({position_name})", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run PCA analysis on multiple data classes")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['intern', 'llava_next', 'qwen'],
                       help='Model name')
    parser.add_argument('--classes', type=str, default=None,
                       help='Comma-separated class combinations (e.g., "image:laion_coco:safe,text:safeguard:unsafe"). If not specified, use all classes.')
    parser.add_argument('--max_samples_per_class', type=int, default=100,
                       help='Maximum samples per class (None for all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/pca)')
    args = parser.parse_args()
    
    # Parse selected classes
    selected_classes = None
    if args.classes:
        class_strings = [s.strip() for s in args.classes.split(',')]
        selected_classes = [parse_class_string(cs) for cs in class_strings]
        print(f"Selected classes: {selected_classes}")
    
    # Set default output directory
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'pca')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    # Get norm layer and number of layers
    norm = find_norm(model)
    num_layers = find_num_hidden_layers(model)
    print(f"Model has {num_layers} layers")
    
    # Load data classes
    print("Loading data classes...")
    data_classes = load_data_classes(selected_classes, args.max_samples_per_class)
    
    if not data_classes:
        print("No data classes loaded. Exiting.")
        return
    
    # Collect all samples with class information
    all_samples = []
    all_class_labels = []  # Integer labels for PCA coloring
    all_class_names = []   # String names for legend
    
    class_to_label = {}
    label_counter = 0
    
    for modality, type_name, safety, samples in data_classes:
        class_key = f"{modality}:{type_name}:{safety}"
        if class_key not in class_to_label:
            class_to_label[class_key] = label_counter
            label_counter += 1
        
        label = class_to_label[class_key]
        for sample in samples:
            all_samples.append(sample)
            all_class_labels.append(label)
            all_class_names.append(class_key)
    
    print(f"Total samples: {len(all_samples)}")
    
    # Extract hidden states
    print("Extracting hidden states...")
    pre_inst_vectors_all_layers = [[] for _ in range(num_layers)]
    last_token_vectors_all_layers = [[] for _ in range(num_layers)]
    
    successful_labels = []
    successful_names = []
    sample_idx = 0
    
    for sample in tqdm(all_samples, desc="Processing samples"):
        try:
            pre_inst_vecs, last_token_vecs = extract_hidden_states_at_positions(
                model, processor, args.model_name,
                sample['prompt'], sample.get('image_path'),
                norm, num_layers
            )
            
            if pre_inst_vecs is not None and last_token_vecs is not None:
                for layer_idx in range(num_layers):
                    pre_inst_vectors_all_layers[layer_idx].append(pre_inst_vecs[layer_idx])
                    last_token_vectors_all_layers[layer_idx].append(last_token_vecs[layer_idx])
                
                successful_labels.append(all_class_labels[sample_idx])
                successful_names.append(all_class_names[sample_idx])
        except Exception:
            # Skip this sample if any error occurs
            pass
        
        sample_idx += 1
    
    print(f"Successfully extracted hidden states for {len(successful_labels)} samples")
    
    # Generate PCA plots for all layers in one plot
    print("Generating PCA plots...")
    
    # Pre-instruction position - all layers in one plot
    if any(pre_inst_vectors_all_layers):
        save_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_all_layers_pre_instruction_pca.png"
        )
        create_pca_plot_all_layers(
            pre_inst_vectors_all_layers,
            successful_labels,
            successful_names,
            "pre_instruction",
            save_path
        )
        print(f"Saved: {save_path}")
    
    # Last token position - all layers in one plot
    if any(last_token_vectors_all_layers):
        save_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_all_layers_last_token_pca.png"
        )
        create_pca_plot_all_layers(
            last_token_vectors_all_layers,
            successful_labels,
            successful_names,
            "last_token",
            save_path
        )
        print(f"Saved: {save_path}")
    
    print(f"✅ All PCA plots saved to: {args.output_dir}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
