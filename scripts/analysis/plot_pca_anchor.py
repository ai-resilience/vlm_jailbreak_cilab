#!/usr/bin/env python3
"""PCA analysis with anchor space projection.

Uses "text:safeguard:safe,text:safeguard:unsafe" as anchor to build PCA space,
then projects other classes into this anchor space for visualization.
"""
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

# Anchor classes (used to build PCA space)
ANCHOR_CLASSES = [
    ('text', 'safeguard', 'safe'),
    ('text', 'safeguard', 'unsafe'),
]


def parse_class_string(class_str: str) -> Tuple[str, str, str]:
    """Parse class string 'modality:type:safety' into tuple."""
    parts = class_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid class format: {class_str}. Expected 'modality:type:safety'")
    return tuple(parts)


def load_laion_coco_images(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load LAION-COCO images from directory."""
    if not os.path.exists(data_dir):
        print(f"Warning: Directory not found: {data_dir}")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
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
    """Load LLMSafeguard texts from JSONL file."""
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
    """Load VLGuard images from directory."""
    if not os.path.exists(image_dir):
        print(f"Warning: Directory not found: {image_dir}")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
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
    """Load selected data classes with metadata."""
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
    """Get instruction token for a model."""
    inst_tokens = {
        'intern': '<|im_end|>\n<|im_start|>assistant\n',
        'llava_next': '[/INST]',
        'qwen': '<|im_end|>\n<|im_start|>assistant',
    }
    return inst_tokens.get(model_name)


def get_pre_instruction_position(model_name: str, tokenizer: Any, input_ids: torch.Tensor) -> Optional[int]:
    """Find token position right before instruction tokens."""
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    
    inst_token = get_instruction_token(model_name)
    if not inst_token:
        return None
    
    try:
        tokenized_inst = tokenizer(inst_token, return_tensors='pt', add_special_tokens=False)
        inst_token_ids = tokenized_inst.input_ids[0]
        inst_token_ids = inst_token_ids.to(input_ids.device)
        inst_token_len = len(inst_token_ids)
    except:
        return None
    
    seq_len = len(input_ids)
    for i in range(seq_len - inst_token_len + 1):
        if torch.equal(input_ids[i:i+inst_token_len], inst_token_ids):
            return i - 1
    
    return None


def get_input_ids_for_intern(model: Any, processor: Any, prompt: str, image_path: Optional[str]) -> Optional[torch.Tensor]:
    """Get input_ids for intern model."""
    try:
        from src.models.intern.preprocess import load_image
        
        prompt_text = prompt
        if image_path is not None:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(model.device)
        else:
            pixel_values = None
        
        if pixel_values is not None and '<image>' not in prompt_text:
            prompt_text = '<image>\n' + prompt_text
        
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        model.img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        template = model.conv_template.copy()
        template.system_message = model.system_message
        template.append_message(template.roles[0], prompt_text)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = processor(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(model.device)
        
        return input_ids
    except Exception:
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
    """Extract hidden states at pre-instruction and last token positions."""
    try:
        hidden_states = extract_hidden_states(model, processor, model_name, prompt, image_path, build_prompt)
        last_token_vectors = extract_token_hidden_states(hidden_states, norm_layer, token_index=-1, num_layers=num_layers)
        
        input_ids = None
        if model_name == "intern":
            input_ids = get_input_ids_for_intern(model, processor, prompt, image_path)
        else:
            try:
                inputs, attention_mask = build_prompt(model, processor, model_name, image_path, prompt)
                
                if model_name == "deepseek2":
                    input_ids = attention_mask.input_ids if hasattr(attention_mask, 'input_ids') else None
                elif isinstance(inputs, dict):
                    input_ids = inputs.get('input_ids', None)
                elif hasattr(inputs, 'input_ids'):
                    input_ids = inputs.input_ids
            except Exception:
                input_ids = None
        
        pre_inst_vectors = None
        if input_ids is not None:
            try:
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                pre_inst_pos = get_pre_instruction_position(model_name, tokenizer, input_ids)
                
                if pre_inst_pos is not None:
                    pre_inst_vectors = extract_token_hidden_states(hidden_states, norm_layer, token_index=pre_inst_pos, num_layers=num_layers)
            except Exception:
                pass
        
        return pre_inst_vectors, last_token_vectors
    
    except Exception:
        return None, None


def create_pca_plot_all_layers_anchor(
    anchor_vectors_all_layers: List[List[np.ndarray]],
    anchor_labels: List[int],
    anchor_names: List[str],
    projected_vectors_all_layers: List[List[np.ndarray]],
    projected_labels: List[int],
    projected_names: List[str],
    position_name: str,
    save_path: str
) -> None:
    """Create PCA plot with anchor space projection.
    
    Args:
        anchor_vectors_all_layers: Anchor vectors per layer [layer][sample][dim]
        anchor_labels: Anchor class labels
        anchor_names: Anchor class names
        projected_vectors_all_layers: Projected vectors per layer [layer][sample][dim]
        projected_labels: Projected class labels
        projected_names: Projected class names
        position_name: Position name ('pre_instruction' or 'last_token')
        save_path: Path to save figure
    """
    num_layers = len(anchor_vectors_all_layers)
    if num_layers == 0:
        print("Warning: No layers to plot")
        return
    
    grid_size = int(np.ceil(np.sqrt(num_layers)))
    
    # Get all unique classes (anchor + projected)
    all_labels = anchor_labels + projected_labels
    unique_classes = sorted(set(all_labels))
    num_classes = len(unique_classes)
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_classes)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_classes)}
    marker_map = {label: markers[i % len(markers)] for i, label in enumerate(unique_classes)}
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Build PCA space from anchor data and project all data
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        if not anchor_vectors_all_layers[layer_idx]:
            ax.axis('off')
            continue
        
        # Stack anchor vectors
        anchor_vecs = np.stack(anchor_vectors_all_layers[layer_idx])  # [num_anchor_samples, dim]
        
        # Fit PCA on anchor data
        pca = PCA(n_components=2)
        pca.fit(anchor_vecs)
        
        # Transform anchor data
        anchor_2d = pca.transform(anchor_vecs)
        
        # Plot anchor classes
        anchor_unique = sorted(set(anchor_labels))
        for class_label in anchor_unique:
            idxs = np.where(np.array(anchor_labels) == class_label)[0]
            if len(idxs) == 0:
                continue
            
            class_name = anchor_names[idxs[0]] if idxs[0] < len(anchor_names) else f"Class {class_label}"
            
            ax.scatter(
                anchor_2d[idxs, 0], anchor_2d[idxs, 1],
                label=class_name if layer_idx == 0 else "",
                alpha=0.8,
                s=40,
                c=[color_map[class_label]],
                marker=marker_map[class_label],
                edgecolors='black',
                linewidths=0.5
            )
        
        # Project and plot projected classes
        if projected_vectors_all_layers[layer_idx]:
            projected_vecs = np.stack(projected_vectors_all_layers[layer_idx])  # [num_projected_samples, dim]
            projected_2d = pca.transform(projected_vecs)
            
            projected_unique = sorted(set(projected_labels))
            for class_label in projected_unique:
                idxs = np.where(np.array(projected_labels) == class_label)[0]
                if len(idxs) == 0:
                    continue
                
                class_name = projected_names[idxs[0]] if idxs[0] < len(projected_names) else f"Class {class_label}"
                
                ax.scatter(
                    projected_2d[idxs, 0], projected_2d[idxs, 1],
                    label=class_name if layer_idx == 0 else "",
                    alpha=0.6,
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
    all_names = anchor_names + projected_names
    for label in unique_classes:
        label_idx = None
        for i, lbl in enumerate(all_labels):
            if lbl == label:
                label_idx = i
                break
        
        class_name = all_names[label_idx] if label_idx is not None and label_idx < len(all_names) else f"Class {label}"
        
        handles.append(
            plt.Line2D([0], [0], marker=marker_map[label], color='w',
                      label=class_name,
                      markerfacecolor=color_map[label], markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5)
        )
    fig.legend(handles=handles, loc='lower center', ncol=min(5, len(unique_classes)), 
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f"2D PCA - Anchor Space Projection ({position_name})", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run PCA analysis with anchor space projection")
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['intern', 'llava_next', 'qwen'],
                       help='Model name')
    parser.add_argument('--max_samples_per_class', type=int, default=100,
                       help='Maximum samples per class (None for all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../result/pca)')
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        args.output_dir = str(project_root.parent / 'result' / 'pca')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, processor, tokenizer = load_model(args.model_name)
    model.eval()
    
    norm = find_norm(model)
    num_layers = find_num_hidden_layers(model)
    print(f"Model has {num_layers} layers")
    
    # Load anchor classes (text:safeguard:safe, text:safeguard:unsafe)
    print("Loading anchor classes...")
    anchor_data_classes = load_data_classes(ANCHOR_CLASSES, args.max_samples_per_class)
    
    # Load all other classes for projection
    all_classes = [
        # ('image', 'laion_coco', 'safe'),
        # ('image', 'laion_coco', 'unsafe'),
        # ('text', 'safeguard', 'safe'),
        # ('text', 'safeguard', 'unsafe'),
        ('image', 'figstep', 'safe'),
        ('image', 'figstep', 'unsafe'),
        ('text', 'figstep', 'safe'),
        ('text', 'figstep', 'unsafe'),
        # ('image', 'ocr', 'unsafe'),
        # ('image', 'VLGuard', 'unsafe'),
    ]
    projected_classes = [cls for cls in all_classes if cls not in ANCHOR_CLASSES]
    
    print("Loading projected classes...")
    projected_data_classes = load_data_classes(projected_classes, args.max_samples_per_class)
    
    if not anchor_data_classes:
        print("Error: No anchor classes loaded. Exiting.")
        return
    
    # Collect anchor samples
    anchor_samples = []
    anchor_labels = []
    anchor_names = []
    
    class_to_label = {}
    label_counter = 0
    
    for modality, type_name, safety, samples in anchor_data_classes:
        class_key = f"{modality}:{type_name}:{safety}"
        if class_key not in class_to_label:
            class_to_label[class_key] = label_counter
            label_counter += 1
        
        label = class_to_label[class_key]
        for sample in samples:
            anchor_samples.append(sample)
            anchor_labels.append(label)
            anchor_names.append(class_key)
    
    # Collect projected samples
    projected_samples = []
    projected_labels = []
    projected_names = []
    
    for modality, type_name, safety, samples in projected_data_classes:
        class_key = f"{modality}:{type_name}:{safety}"
        if class_key not in class_to_label:
            class_to_label[class_key] = label_counter
            label_counter += 1
        
        label = class_to_label[class_key]
        for sample in samples:
            projected_samples.append(sample)
            projected_labels.append(label)
            projected_names.append(class_key)
    
    print(f"Anchor samples: {len(anchor_samples)}")
    print(f"Projected samples: {len(projected_samples)}")
    
    # Extract hidden states for anchor samples
    print("Extracting hidden states for anchor samples...")
    anchor_pre_inst_vectors_all_layers = [[] for _ in range(num_layers)]
    anchor_last_token_vectors_all_layers = [[] for _ in range(num_layers)]
    
    anchor_successful_labels = []
    anchor_successful_names = []
    sample_idx = 0
    
    for sample in tqdm(anchor_samples, desc="Processing anchor samples"):
        try:
            pre_inst_vecs, last_token_vecs = extract_hidden_states_at_positions(
                model, processor, args.model_name,
                sample['prompt'], sample.get('image_path'),
                norm, num_layers
            )
            
            if pre_inst_vecs is not None and last_token_vecs is not None:
                for layer_idx in range(num_layers):
                    anchor_pre_inst_vectors_all_layers[layer_idx].append(pre_inst_vecs[layer_idx])
                    anchor_last_token_vectors_all_layers[layer_idx].append(last_token_vecs[layer_idx])
                
                anchor_successful_labels.append(anchor_labels[sample_idx])
                anchor_successful_names.append(anchor_names[sample_idx])
        except Exception:
            pass
        
        sample_idx += 1
    
    print(f"Successfully extracted {len(anchor_successful_labels)} anchor samples")
    
    # Extract hidden states for projected samples
    print("Extracting hidden states for projected samples...")
    projected_pre_inst_vectors_all_layers = [[] for _ in range(num_layers)]
    projected_last_token_vectors_all_layers = [[] for _ in range(num_layers)]
    
    projected_successful_labels = []
    projected_successful_names = []
    sample_idx = 0
    
    for sample in tqdm(projected_samples, desc="Processing projected samples"):
        try:
            pre_inst_vecs, last_token_vecs = extract_hidden_states_at_positions(
                model, processor, args.model_name,
                sample['prompt'], sample.get('image_path'),
                norm, num_layers
            )
            
            if pre_inst_vecs is not None and last_token_vecs is not None:
                for layer_idx in range(num_layers):
                    projected_pre_inst_vectors_all_layers[layer_idx].append(pre_inst_vecs[layer_idx])
                    projected_last_token_vectors_all_layers[layer_idx].append(last_token_vecs[layer_idx])
                
                projected_successful_labels.append(projected_labels[sample_idx])
                projected_successful_names.append(projected_names[sample_idx])
        except Exception:
            pass
        
        sample_idx += 1
    
    print(f"Successfully extracted {len(projected_successful_labels)} projected samples")
    
    # Generate PCA plots
    print("Generating PCA plots with anchor space projection...")
    
    # Pre-instruction position
    if any(anchor_pre_inst_vectors_all_layers):
        save_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_anchor_pre_instruction_pca.png"
        )
        create_pca_plot_all_layers_anchor(
            anchor_pre_inst_vectors_all_layers,
            anchor_successful_labels,
            anchor_successful_names,
            projected_pre_inst_vectors_all_layers,
            projected_successful_labels,
            projected_successful_names,
            "pre_instruction",
            save_path
        )
        print(f"Saved: {save_path}")
    
    # Last token position
    if any(anchor_last_token_vectors_all_layers):
        save_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_anchor_last_token_pca.png"
        )
        create_pca_plot_all_layers_anchor(
            anchor_last_token_vectors_all_layers,
            anchor_successful_labels,
            anchor_successful_names,
            projected_last_token_vectors_all_layers,
            projected_successful_labels,
            projected_successful_names,
            "last_token",
            save_path
        )
        print(f"Saved: {save_path}")
    
    print(f"✅ All PCA plots saved to: {args.output_dir}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
