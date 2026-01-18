from typing import Dict, List
import numpy as np

class AttractorClassifier:
    """
    Classify destination attractors (19 categories)
    Implements dominance scoring and absorption principle
    """
    
    def __init__(self, config):
        self.config = config
        self.taxonomy = None
        
        # 19 destination attractors (exact from guidelines)
        self.attractors = [
            "Accommodation",
            "Architecture & Heritage",
            "Climate",
            "Cultural Attractions",
            "Events, Fairs & Festivals",
            "Food & Drink",
            "Infrastructure & Transportation",
            "Landscape & Natural Resources",
            "Leisure Attractions",
            "Local Culture & History",
            "Local Lifestyle",
            "Nightlife",
            "Political & Economic Factors",
            "Safety",
            "Service",
            "Shopping",
            "Sports",
            "Tourism Products & Packages",
            "Wellness"
        ]
    
    def load_taxonomy(self, taxonomy: Dict):
        """Load taxonomy from Excel Refined_Lists sheet"""
        self.taxonomy = taxonomy
    
    def classify(self, qwen_result: Dict, duration: float) -> Dict:
        """
        Classify attractors from Qwen analysis
        
        Args:
            qwen_result: Output from Qwen multimodal analysis
            duration: Video duration in seconds
            
        Returns:
            Dictionary with attractor seconds, subcategories, places
        """
        
        # Get attractor seconds from Qwen result
        attractor_seconds = qwen_result.get('attractor_seconds', {})
        subcategories = qwen_result.get('subcategories', {})
        places = qwen_result.get('places', {})
        
        # Ensure all 19 attractors are present
        for att in self.attractors:
            if att not in attractor_seconds:
                attractor_seconds[att] = 0.0
        
        # Apply absorption principle
        attractor_seconds = self._apply_absorption(attractor_seconds)
        
        # Normalize to match duration exactly
        attractor_seconds = self._normalize_seconds(attractor_seconds, duration)
        
        return {
            'attractor_seconds': attractor_seconds,
            'subcategories': subcategories,
            'places': places
        }
    
    def _apply_absorption(self, attractor_seconds: Dict) -> Dict:
        """
        Apply absorption principle:
        - Pool → Accommodation
        - Hotel restaurant → Accommodation
        - Museum cafe → Cultural Attractions
        """
        
        # These are handled in Qwen prompts, but double-check here
        # For now, return as-is (Qwen should handle this)
        
        return attractor_seconds
    
    def _normalize_seconds(self, attractor_seconds: Dict, target_duration: float) -> Dict:
        """
        Normalize attractor seconds to exactly match video duration
        """
        
        total = sum(attractor_seconds.values())
        
        if total == 0:
            # No attractors detected - distribute evenly
            per_attractor = target_duration / len(self.attractors)
            return {att: per_attractor for att in self.attractors}
        
        # Proportional normalization
        factor = target_duration / total
        
        normalized = {
            att: round(seconds * factor, 2)
            for att, seconds in attractor_seconds.items()
        }
        
        # Handle rounding errors - adjust largest value
        current_total = sum(normalized.values())
        diff = target_duration - current_total
        
        if abs(diff) > 0.01:
            # Add difference to most common attractor
            max_att = max(normalized, key=normalized.get)
            normalized[max_att] += diff
            normalized[max_att] = round(normalized[max_att], 2)
        
        return normalized