"""
QWEN-VL MULTIMODAL ANALYZER
Core AI analysis using Qwen-2.5-VL for frame-by-frame video understanding
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
from typing import List, Dict
import json
import base64
from io import BytesIO

class QwenAnalyzer:
    """
    Qwen-2.5-VL multimodal analysis for tourism video coding
    Handles frame classification into attractors and narrative frames
    """
    
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Qwen-VL on {self.device}...")
        
        # Load Qwen model
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load taxonomy for prompt engineering
        self.taxonomy = None
        
        print("✓ Qwen-VL model loaded")
    
    def load_taxonomy(self, taxonomy: Dict):
        """Load the 19 attractor taxonomy"""
        self.taxonomy = taxonomy
        
    async def analyze_video(
        self, 
        frames: List[Dict],
        audio_transcript: str,
        ocr_text: Dict,
        duration: float
    ) -> Dict:
        """
        Main analysis function for complete video
        
        Args:
            frames: List of extracted frames with timestamps
            audio_transcript: Whisper transcription
            ocr_text: Timeline of on-screen text
            duration: Total video duration in seconds
            
        Returns:
            Complete analysis with attractors and frames
        """
        
        # Build master prompt
        master_prompt = self._build_master_prompt(
            duration, audio_transcript, ocr_text
        )
        
        # Process frames in batches (8-10 frames per API call)
        batch_size = 8
        all_results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            # Analyze batch
            batch_result = await self._analyze_frame_batch(
                batch, master_prompt
            )
            
            all_results.extend(batch_result)
        
        # Aggregate frame-level results into video-level
        video_result = self._aggregate_frame_results(all_results, duration)
        
        return video_result
    
    def _build_master_prompt(
        self, 
        duration: float, 
        transcript: str, 
        ocr_timeline: Dict
    ) -> str:
        """
        Build comprehensive analysis prompt
        Following the exact guidelines from your documents
        """
        
        # Extract attractor list
        attractor_list = self._get_attractor_list()
        
        # Extract frame indicators
        frame_indicators = self._get_frame_indicators()
        
        prompt = f"""You are a research-grade video analyst specializing in tourism content analysis for academic research.

CRITICAL RULES (MANDATORY):
1. Use ONLY these 19 destination attractors: {attractor_list}
2. Each second must belong to exactly ONE primary attractor
3. Apply absorption principle: pools→Accommodation, hotel restaurants→Accommodation
4. Total attractor seconds MUST equal {duration} seconds
5. Classify narrative frames independently for Visual/Audio/Text modalities
6. Each modality's frame seconds (AE+LI+IN+CR) MUST equal {duration} seconds

VIDEO METADATA:
- Duration: {duration} seconds
- Audio transcript: "{transcript[:500]}..."
- Has on-screen text: {len(ocr_timeline) > 0}

TASK 1: DESTINATION ATTRACTOR CLASSIFICATION
For each frame timestamp, identify the PRIMARY destination attractor.
Use dominance scoring: object_size × confidence × centrality

Choose from these 19 categories ONLY:
{self._format_attractor_taxonomy()}

TASK 2: NARRATIVE FRAME CLASSIFICATION
Classify each second across THREE INDEPENDENT modalities:

VISUAL FRAME (cinematography, composition, creator presence):
{frame_indicators['visual']}

AUDIO FRAME (soundtrack, voiceover, sound design):
{frame_indicators['audio']}

TEXT FRAME (overlay text content and style):
{frame_indicators['text']}

OUTPUT FORMAT (JSON ONLY, NO EXPLANATION):
{{
  "frame_timestamp": "00:05-00:07",
  "attractor": "Architecture & Heritage",
  "subcategory": "Monuments",
  "place": "Pyramids of Giza",
  "confidence": 0.92,
  "visual_frame": "AE",
  "visual_tag": "Cinematic composition",
  "audio_frame": "LI",
  "audio_tag": "Creator voiceover narration",
  "text_frame": "IN",
  "text_tag": "Lists or enumerations"
}}

Analyze each frame and provide structured JSON output."""

        return prompt
    
    async def _analyze_frame_batch(
        self, 
        frames: List[Dict], 
        prompt: str
    ) -> List[Dict]:
        """
        Analyze a batch of frames using Qwen-VL
        """
        
        # Prepare images
        images = [Image.fromarray(frame['frame']) for frame in frames]
        
        # Build conversation with multiple images
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image", "image": img} for img in images]
                ]
            }
        ]
        
        # Prepare for the model
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,  # Deterministic
                temperature=0.0
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )[0]
        
        # Parse JSON response
        results = self._parse_qwen_response(generated_text, frames)
        
        return results
    
    def _parse_qwen_response(
        self, 
        response: str, 
        frames: List[Dict]
    ) -> List[Dict]:
        """
        Parse Qwen's JSON response into structured data
        """
        try:
            # Extract JSON from response
            # Qwen might wrap it in markdown or add explanation
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
            
            json_str = response[json_start:json_end]
            
            # Parse
            parsed = json.loads(json_str)
            
            # If single object, wrap in list
            if isinstance(parsed, dict):
                parsed = [parsed]
            
            # Map to frame timestamps
            for i, result in enumerate(parsed):
                if i < len(frames):
                    result['timestamp'] = frames[i]['timestamp']
                    result['frame_number'] = frames[i]['frame_number']
            
            return parsed
            
        except Exception as e:
            print(f"Warning: Failed to parse Qwen response: {e}")
            # Fallback: return empty results
            return [{
                'timestamp': frame['timestamp'],
                'frame_number': frame['frame_number'],
                'error': 'parse_failed'
            } for frame in frames]
    
    def _aggregate_frame_results(
        self, 
        frame_results: List[Dict], 
        duration: float
    ) -> Dict:
        """
        Aggregate frame-level results into video-level metrics
        """
        
        # Initialize counters
        attractor_seconds = {att: 0.0 for att in self._get_attractor_list().split(', ')}
        
        narrative_visual = {'AE': 0.0, 'LI': 0.0, 'IN': 0.0, 'CR': 0.0}
        narrative_audio = {'AE': 0.0, 'LI': 0.0, 'IN': 0.0, 'CR': 0.0}
        narrative_text = {'AE': 0.0, 'LI': 0.0, 'IN': 0.0, 'CR': 0.0}
        
        tags_visual = {'AE': set(), 'LI': set(), 'IN': set(), 'CR': set()}
        tags_audio = {'AE': set(), 'LI': set(), 'IN': set(), 'CR': set()}
        tags_text = {'AE': set(), 'LI': set(), 'IN': set(), 'CR': set()}
        
        subcategories = {}
        places = {}
        
        # Process each frame result
        for result in frame_results:
            if 'error' in result:
                continue
            
            # Time window for this frame (approx 2 seconds)
            time_window = 2.0
            
            # Attractors
            attractor = result.get('attractor')
            if attractor and attractor in attractor_seconds:
                attractor_seconds[attractor] += time_window
                
                # Track subcategory and place
                if 'subcategory' in result:
                    subcategories[attractor] = result['subcategory']
                if 'place' in result and result['place']:
                    places[attractor] = result['place']
            
            # Narrative frames
            visual_frame = result.get('visual_frame')
            if visual_frame in narrative_visual:
                narrative_visual[visual_frame] += time_window
                if 'visual_tag' in result:
                    tags_visual[visual_frame].add(result['visual_tag'])
            
            audio_frame = result.get('audio_frame')
            if audio_frame in narrative_audio:
                narrative_audio[audio_frame] += time_window
                if 'audio_tag' in result:
                    tags_audio[audio_frame].add(result['audio_tag'])
            
            text_frame = result.get('text_frame')
            if text_frame in narrative_text:
                narrative_text[text_frame] += time_window
                if 'text_tag' in result:
                    tags_text[text_frame].add(result['text_tag'])
        
        # Normalize to exactly match duration
        attractor_total = sum(attractor_seconds.values())
        if attractor_total > 0:
            factor = duration / attractor_total
            attractor_seconds = {k: v * factor for k, v in attractor_seconds.items()}
        
        # Normalize narrative frames
        for modality in [narrative_visual, narrative_audio, narrative_text]:
            total = sum(modality.values())
            if total > 0:
                factor = duration / total
                for key in modality:
                    modality[key] *= factor
        
        # Get dominant tags (most frequent)
        dominant_tags_visual = {k: list(v)[0] if v else "" for k, v in tags_visual.items()}
        dominant_tags_audio = {k: list(v)[0] if v else "" for k, v in tags_audio.items()}
        dominant_tags_text = {k: list(v)[0] if v else "" for k, v in tags_text.items()}
        
        return {
            'attractor_seconds': attractor_seconds,
            'subcategories': subcategories,
            'places': places,
            'narrative_visual': narrative_visual,
            'narrative_audio': narrative_audio,
            'narrative_text': narrative_text,
            'tags_visual': dominant_tags_visual,
            'tags_audio': dominant_tags_audio,
            'tags_text': dominant_tags_text,
            'frame_results': frame_results  # Keep raw for debugging
        }
    
    def _get_attractor_list(self) -> str:
        """Get comma-separated list of 19 attractors"""
        return "Accommodation, Architecture & Heritage, Climate, Cultural Attractions, Events Fairs & Festivals, Food & Drink, Infrastructure & Transportation, Landscape & Natural Resources, Leisure Attractions, Local Culture & History, Local Lifestyle, Nightlife, Political & Economic Factors, Safety, Service, Shopping, Sports, Tourism Products & Packages, Wellness"
    
    def _format_attractor_taxonomy(self) -> str:
        """Format taxonomy for prompt"""
        attractors = self._get_attractor_list().split(', ')
        return '\n'.join([f"- {att}" for att in attractors])
    
    def _get_frame_indicators(self) -> Dict:
        """Get narrative frame indicators"""
        return {
            'visual': """
AE (Aesthetic-Hedonic): Cinematic composition, Golden hour lighting, Drone views, Slow pans, Symmetrical framing
LI (Lifestyle-Identity): Selfie-style, Creator face centered, POV walking, OOTD showcase, Mirror selfies
IN (Inspirational-Narrative): Sequential storytelling, Focus on objects/locations, Educational B-roll, Detail shots
CR (Critical-Reflexive): Talking-to-camera critique, Juxtaposition, Evidence documentation, Problem framing
""",
            'audio': """
AE (Aesthetic-Hedonic): Instrumental trending audio, Cinematic orchestral, Ambient soundscape, Lo-fi beats
LI (Lifestyle-Identity): Creator voiceover narration, Casual tone, Personal anecdotes, GRWM-style narration
IN (Inspirational-Narrative): Explanatory voiceover, Calm instructional tone, Documentary-style, Guide narration
CR (Critical-Reflexive): Direct spoken critique, Serious/confrontational tone, Sarcastic audio, Warning tone
""",
            'text': """
AE (Aesthetic-Hedonic): Minimal text (≤5 words), Beautiful/stunning adjectives, Emoji-only overlays
LI (Lifestyle-Identity): First-person language (I/my/me), POV framing, Lifestyle hashtags, Personal journey
IN (Inspirational-Narrative): Lists/enumerations, Instructional phrasing, Contextual information, How-to language
CR (Critical-Reflexive): Explicit claims, Warning language (don't/beware/scam), Expectation vs reality
"""
        }