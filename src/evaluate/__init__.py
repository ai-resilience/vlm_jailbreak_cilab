"""Evaluation metrics for VLM safety evaluation."""
from .base import (
    BaseMetric,
    load_entries,
    write_text_result,
    format_rr_report,
    format_refusal_report,
    format_unsafe_report,
    refusal_rate_from_field,
    refusal_rate_from_flags,
    unsafe_rate_from_flags,
    is_refusal_keyword,
)
from .keyword import KeywordMetric
from .llamaguard import LlamaGuardMetric
from .beaverdam import BeaverDamMetric
from .wildguard import WildGuardMetric
from typing import List, Dict, Tuple, Optional, Any
from collections import OrderedDict
import json
import os


def load_metric(metric_name: str, **kwargs) -> BaseMetric:
    """Factory function to load metrics by name.
    
    Args:
        metric_name: Name of the metric ('keyword', 'llamaguard4')
        **kwargs: Additional metric-specific arguments
        
    Returns:
        Metric instance
    """
    metric_map = {
        'keyword': KeywordMetric,
        # Only LlamaGuard v4 is exposed; keep only explicit key
        'llamaguard4': lambda **kw: LlamaGuardMetric(**kw),
        'beaverdam': lambda **kw: BeaverDamMetric(**kw),
        'wildguard': lambda **kw: WildGuardMetric(**kw),
    }
    
    if metric_name not in metric_map:
        raise ValueError(
            f"Unknown metric: {metric_name}. Available metrics: {list(metric_map.keys())}"
        )
    
    metric_class = metric_map[metric_name]
    
    if callable(metric_class) and not isinstance(metric_class, type):
        # It's a lambda function
        metric = metric_class(**kwargs)
    else:
        metric = metric_class(**kwargs)
    
    return metric


def evaluate_with_metric(
    metric_name: str,
    entries: List[Dict],
    dataset_name: Optional[str] = None,
    is_image: bool = False,
    image_dataset: Optional[str] = None,
    **metric_kwargs
) -> Tuple[List[bool], float]:
    """Evaluate entries using a metric.
    
    Args:
        metric_name: Name of the metric
        entries: List of entries to evaluate
        dataset_name: Name of the dataset (for special handling in LlamaGuard)
        is_image: Whether the dataset contains images (for LlamaGuard)
        image_dataset: Name of the image dataset (for mm_safety datasets in LlamaGuard)
        **metric_kwargs: Additional metric-specific arguments
        
    Returns:
        Tuple of (flags: List[bool], rate: float)
    """
    metric = load_metric(metric_name, **metric_kwargs)
    metric.load()
    
    # Pass special parameters to LlamaGuard evaluate method
    if metric_name == 'llamaguard4':
        from .llamaguard import LlamaGuardMetric
        if isinstance(metric, LlamaGuardMetric):
            return metric.evaluate(entries, dataset_name=dataset_name, is_image=is_image, image_dataset=image_dataset)
    
    # BeaverDam also supports the same interface
    if metric_name == 'beaverdam':
        return metric.evaluate(entries, dataset_name=dataset_name, is_image=is_image, image_dataset=image_dataset)
    
    # WildGuard also supports the same interface
    if metric_name == 'wildguard':
        return metric.evaluate(entries, dataset_name=dataset_name, is_image=is_image, image_dataset=image_dataset)
    
    return metric.evaluate(entries)


def save_evaluation_results(
    entries: List[Dict],
    flags: List[bool],
    output_path: str,
    metric_name: str,
    result_field: str = "refusal"
) -> str:
    """Save evaluation results to JSONL file.
    
    Args:
        entries: Original entries
        flags: Boolean flags from evaluation
        output_path: Output JSONL file path
        metric_name: Name of the metric used
        result_field: Field name for the result (default: 'refusal')
        
    Returns:
        Path to the saved file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for entry, flag in zip(entries, flags):
            ordered = OrderedDict()
            ordered[result_field] = str(flag)
            
            # Preserve original fields
            for key in ['prompt', 'response', 'label', 'image']:
                if key in entry:
                    ordered[key] = entry[key]
            
            fout.write(json.dumps(ordered, ensure_ascii=False) + '\n')
    
    return output_path


__all__ = [
    'BaseMetric',
    'KeywordMetric',
    'LlamaGuardMetric',
    'BeaverDamMetric',
    'WildGuardMetric',
    'load_metric',
    'evaluate_with_metric',
    'save_evaluation_results',
    'load_entries',
    'write_text_result',
    'format_rr_report',
    'format_refusal_report',
    'format_unsafe_report',
    'refusal_rate_from_field',
    'refusal_rate_from_flags',
    'unsafe_rate_from_flags',
    'is_refusal_keyword',
]

