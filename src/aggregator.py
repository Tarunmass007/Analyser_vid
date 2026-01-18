from typing import Dict
from datetime import datetime

class ResultAggregator:
    """
    Aggregate all analysis results into final output format
    """
    
    def __init__(self, config):
        self.config = config
    
    def aggregate(
        self,
        video_id: str,
        video_path: str,
        technical: Dict,
        attractors: Dict,
        narratives: Dict
    ) -> Dict:
        """
        Aggregate all results into single dictionary for Excel writing
        
        Returns:
            Complete result dictionary matching Excel structure
        """
        
        result = {
            # Metadata
            'video_id': video_id,
            'video_path': video_path,
            'video_file_name': technical.get('filename', ''),
            'coding_date': datetime.now().strftime('%Y-%m-%d'),
            
            # Technical
            'duration': technical.get('duration', 0),
            'fps': technical.get('fps', 0),
            'resolution': (technical.get('width', 0), technical.get('height', 0)),
            'audio_type': technical.get('audio_type', ''),
            'text_overlay_count': technical.get('text_overlay_count', 0),
            'total_cuts': technical.get('total_cuts', 0),
            'cuts_per_minute': technical.get('cuts_per_minute', 0),
            
            # Attractors
            'attractor_seconds': attractors.get('attractor_seconds', {}),
            'subcategories': attractors.get('subcategories', {}),
            'places': attractors.get('places', {}),
            
            # Narrative Frames
            'narrative_visual': narratives.get('visual_seconds', {}),
            'narrative_audio': narratives.get('audio_seconds', {}),
            'narrative_text': narratives.get('text_seconds', {}),
            
            'tags_visual': narratives.get('visual_tags', {}),
            'tags_audio': narratives.get('audio_tags', {}),
            'tags_text': narratives.get('text_tags', {}),
            
            'scores_visual': narratives.get('visual_scores', {}),
            'scores_audio': narratives.get('audio_scores', {}),
            'scores_text': narratives.get('text_scores', {}),
            
            # Status
            'status': 'success',
            'processing_time': datetime.now()
        }
        
        return result
