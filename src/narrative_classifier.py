from typing import Dict

class NarrativeClassifier:
    """
    Classify narrative frames: AE/LI/IN/CR across 3 modalities
    Aesthetic-Hedonic, Lifestyle-Identity, Inspirational-Narrative, Critical-Reflexive
    """
    
    def __init__(self, config):
        self.config = config
        self.taxonomy = None
    
    def load_taxonomy(self, taxonomy: Dict):
        """Load frame indicators from taxonomy"""
        self.taxonomy = taxonomy
    
    def classify(self, qwen_result: Dict, audio_result: Dict, ocr_result: Dict, duration: float) -> Dict:
        """
        Classify narrative frames for all 3 modalities
        
        Returns:
            Dictionary with visual/audio/text frame seconds and tags
        """
        
        # Extract from Qwen result
        visual_frames = qwen_result.get('narrative_visual', {'AE': 0, 'LI': 0, 'IN': 0, 'CR': 0})
        audio_frames = qwen_result.get('narrative_audio', {'AE': 0, 'LI': 0, 'IN': 0, 'CR': 0})
        text_frames = qwen_result.get('narrative_text', {'AE': 0, 'LI': 0, 'IN': 0, 'CR': 0})
        
        # Normalize each modality to duration
        visual_frames = self._normalize_frames(visual_frames, duration)
        audio_frames = self._normalize_frames(audio_frames, duration)
        text_frames = self._normalize_frames(text_frames, duration)
        
        # Get dominant tags
        visual_tags = qwen_result.get('tags_visual', {})
        audio_tags = qwen_result.get('tags_audio', {})
        text_tags = qwen_result.get('tags_text', {})
        
        # Calculate scores (0-100)
        visual_scores = self._calculate_scores(visual_frames, duration)
        audio_scores = self._calculate_scores(audio_frames, duration)
        text_scores = self._calculate_scores(text_frames, duration)
        
        return {
            'visual_seconds': visual_frames,
            'visual_tags': visual_tags,
            'visual_scores': visual_scores,
            'audio_seconds': audio_frames,
            'audio_tags': audio_tags,
            'audio_scores': audio_scores,
            'text_seconds': text_frames,
            'text_tags': text_tags,
            'text_scores': text_scores
        }
    
    def _normalize_frames(self, frames: Dict, duration: float) -> Dict:
        """Normalize frame seconds to match duration"""
        
        total = sum(frames.values())
        
        if total == 0:
            # Default to Aesthetic-Hedonic if nothing detected
            return {'AE': duration, 'LI': 0, 'IN': 0, 'CR': 0}
        
        factor = duration / total
        
        normalized = {k: round(v * factor, 2) for k, v in frames.items()}
        
        # Fix rounding
        current_total = sum(normalized.values())
        diff = duration - current_total
        if abs(diff) > 0.01:
            max_frame = max(normalized, key=normalized.get)
            normalized[max_frame] += diff
            normalized[max_frame] = round(normalized[max_frame], 2)
        
        return normalized
    
    def _calculate_scores(self, frames: Dict, duration: float) -> Dict:
        """Calculate percentage scores (0-100)"""
        
        return {
            frame: round((seconds / duration) * 100, 2) if duration > 0 else 0
            for frame, seconds in frames.items()
        }