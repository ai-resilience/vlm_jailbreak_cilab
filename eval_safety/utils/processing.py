import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
# from mathruler.grader import extract_boxed_content, grade_answer
from transformers import AutoProcessor, AutoTokenizer   
import pdb

def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts_qwen(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples for Qwen models"""
    print("Preparing prompts for Qwen")
    prompts = []
    metadata = []


    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue
        
        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # Create prompt
        if args.version == "grpo":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "back":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "hint":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> {args.system_prompt} \n Question: {item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "figstep":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "figstep_usr":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def prepare_prompts_intern(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples for InternVL models"""
    print("Preparing prompts for Intern")
    prompts = []
    metadata = []

    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue

        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue

        if args.version == "figstep":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "figstep_usr":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unknown version: {args.version}")

        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })

        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })

    return prompts, metadata

def prepare_prompts_kimi(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples for Kimi-VL models"""
    print("Preparing prompts for Kimi-VL")
    prompts = []
    metadata = []


    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue

        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        if args.version == "figstep":
            messages = [
                {"role": "system", "content": args.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["question"]},
                    ],
                },
            ]
        elif args.version == "figstep_usr":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["question"] + " " + args.system_prompt},
                    ],
                },
            ]       

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })

        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })

    return prompts, metadata

def process_outputs_simplified(outputs, metadata) -> List[Dict]:
    results = []
    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        meta = metadata[i]
        
        result = {
            "id": meta["id"],
            "question": meta["question"],
            "answer": meta["answer"],
            "prediction": prediction
        }
        results.append(result)
    
    return results
