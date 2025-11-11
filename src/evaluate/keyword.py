"""Keyword-based refusal detection metric."""
from typing import List, Dict, Tuple
from .base import BaseMetric, is_refusal_keyword, refusal_rate_from_flags


class KeywordMetric(BaseMetric):
    """Keyword-based refusal detection metric."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load(self) -> None:
        """No model loading needed for keyword-based metric."""
        pass
    
    def evaluate(self, entries: List[Dict]) -> Tuple[List[bool], float]:
        """Evaluate entries using keyword patterns.
        
        Args:
            entries: List of entries with 'response' field
            
        Returns:
            Tuple of (flags: List[bool], rate: float)
        """
        flags = [is_refusal_keyword(e.get('response')) for e in entries]
        rate = refusal_rate_from_flags(flags)
        return flags, rate

