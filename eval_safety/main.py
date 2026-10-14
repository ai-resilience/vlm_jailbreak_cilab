import argparse
import gc
import json
import os
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from utils.data_loaders import (
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_mathvision_dataset,
    load_hallubench_dataset,
    load_GeoMath_dataset,
    load_Tallyqa_dataset,
    load_MME_dataset,
    load_Figstep_dataset,
    load_mmsafety_typo_dataset,
    load_mmsafety_sd_dataset,
    load_mmsafety_sd_typo_dataset,
)
from utils.processing import (
    prepare_prompts_qwen,
    prepare_prompts_intern,
    prepare_prompts_kimi,
    process_outputs_simplified,
)

SAFETY_DATASETS = {"figstep", "MM-SafetyBench-Typo", "MM-SafetyBench-SD", "MM-SafetyBench-SD-TYPO"}


def convert_result_to_safety_jsonl(input_path: Path, output_path: Path, default_label: int = 0) -> int:
    """Convert result JSON to JSONL format for safety evaluation."""
    with input_path.open("r", encoding="utf-8") as f:
        src = json.load(f)

    results = src.get("results", [])
    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            row = {
                "prompt": item.get("question", ""),
                "response": item.get("prediction", ""),
                "label": default_label,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-family", type=str, required=True, help="Model family for prompt template selection")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.5, help="Top-p sampling")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")
    parser.add_argument("--version", type=str, default="back")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'")
    parser.add_argument("--data-path", type=str, default="eval/eval_data", help="")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of prompts to process per batch (lower = less RAM)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
   
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to evaluate 
    datasets_to_eval = args.datasets.split(",") if args.datasets != "all" else [
        "wemath", "mathvista", "mathverse", "mathvision", "hallubench", "GeoMath", "Tallyqa", "MME", "figstep"
    ]
    
    DATASET_LOADERS = {
        "wemath": load_wemath_dataset,
        "mathvista": load_mathvista_dataset,
        "mathverse": load_mathverse_dataset,
        "mathvision": load_mathvision_dataset,
        "hallubench": load_hallubench_dataset,
        "GeoMath": load_GeoMath_dataset,
        "Tallyqa": load_Tallyqa_dataset,
        "MME": load_MME_dataset,
        "figstep": load_Figstep_dataset,
        "MM-SafetyBench-Typo": load_mmsafety_typo_dataset,
        "MM-SafetyBench-SD": load_mmsafety_sd_dataset,
        "MM-SafetyBench-SD-TYPO": load_mmsafety_sd_typo_dataset,
    }

    # Initialize model
    print(f"Initializing model from {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        trust_remote_code=True
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    BATCH_SIZE = args.batch_size

    PREPARE_FN = {
        "qwen": prepare_prompts_qwen,
        "intern": prepare_prompts_intern,
        "kimi": prepare_prompts_kimi,
    }
    prepare_fn = PREPARE_FN.get(args.model_family)
    if prepare_fn is None:
        raise ValueError(f"Unknown model family: {args.model_family}")

    for dataset_name in datasets_to_eval:
        loader = DATASET_LOADERS.get(dataset_name)
        if loader is None:
            print(f"Unknown dataset: {dataset_name}, skipping.")
            continue

        samples = loader(args.data_path)
        total = len(samples)
        print(f"Loaded {total} samples from {dataset_name}")

        if not samples:
            continue

        all_results = []
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_samples = samples[batch_start:batch_end]

            print(f"  [{dataset_name}] Preparing & generating batch {batch_start}–{batch_end} / {total}")
            prompts, metadata = prepare_fn(dataset_name, batch_samples, args)

            outputs = llm.generate(prompts, sampling_params)
            batch_results = process_outputs_simplified(outputs, metadata)
            all_results.extend(batch_results)

            del batch_samples, prompts, metadata, outputs, batch_results
            gc.collect()

        output_dict = {
            "results": all_results,
            "config": vars(args)
        }
        
        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        if dataset_name in SAFETY_DATASETS:
            jsonl_path = Path(args.output_dir) / f"{dataset_name}.jsonl"
            n = convert_result_to_safety_jsonl(Path(output_path), jsonl_path)
            print(f"  [{dataset_name}] Safety eval JSONL saved: {jsonl_path} ({n} rows)")

        del samples, all_results
        gc.collect()
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()