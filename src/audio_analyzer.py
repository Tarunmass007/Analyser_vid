from typing import Dict, List, Optional
import whisper
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class AudioAnalyzer:
    """
    Audio analysis using OpenAI Whisper
    Transcribes speech and classifies audio type
    """
    
    def __init__(self, config):
        self.config = config
        model_size = config.cfg['models']['whisper_model']
        device = config.device
        
        print(f"Loading Whisper model: {model_size} on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print("✓ Whisper model loaded")
    
    def analyze(self, video_path: str) -> Dict:
        """
        Analyze audio track
        
        Returns:
            Dictionary with type, transcript, language
        """
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                video_path,
                fp16=False,
                language='en',
                task='transcribe'
            )
            
            transcript = result['text'].strip()
            language = result.get('language', 'en')
            
            # Classify audio type based on transcript length
            audio_type = self._classify_audio_type(transcript)
            
            return {
                'type': audio_type,
                'transcript': transcript,
                'language': language,
                'has_speech': len(transcript.split()) > 5,
                'has_music': audio_type in ['Music', 'Music Only', 'Music+Voiceover'],
                'word_count': len(transcript.split())
            }
            
        except Exception as e:
            print(f"  Warning: Audio analysis failed: {e}")
            # Fallback to music-only
            return {
                'type': 'Music',
                'transcript': '',
                'language': 'unknown',
                'has_speech': False,
                'has_music': True,
                'word_count': 0
            }
    
    def _classify_audio_type(self, transcript: str) -> str:
        """
        Classify into 5 types: Music, Voiceover, Music+Voiceover, Ambient, Silent
        Based on transcript word count
        """
        
        word_count = len(transcript.split())
        
        if word_count == 0:
            return "Music"  # Assume music if no words detected
        elif word_count < 10:
            return "Music"  # Very few words = music-driven
        elif word_count < 50:
            return "Music+Voiceover"  # Some narration with music
        else:
            return "VO+Music"  # Significant voiceover
    
    def detect_sentiment(self, transcript: str) -> str:
        """Quick sentiment detection from keywords"""
        
        transcript_lower = transcript.lower()
        
        critical_words = ['bad', 'terrible', 'disappointing', 'avoid', 'scam', 'warning']
        positive_words = ['amazing', 'beautiful', 'perfect', 'love', 'best', 'stunning']
        
        critical_count = sum(1 for word in critical_words if word in transcript_lower)
        positive_count = sum(1 for word in positive_words if word in transcript_lower)
        
        if critical_count > positive_count:
            return "critical"
        elif positive_count > 0:
            return "positive"
        else:
            return "neutral"
