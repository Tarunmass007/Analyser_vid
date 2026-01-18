# --------- COLAB-SAFE OCR WRAPPER ---------
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    print("⚠️ PaddleOCR DISABLED in this environment (Colab-safe mode):", e)
    PADDLE_AVAILABLE = False
# ------------------------------------------

import numpy as np
from typing import List, Dict

class OCRDetector:
    """
    On-screen text detection using PaddleOCR
    Extracts text overlays and captions
    """

    def __init__(self, config):
        self.config = config

        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(lang="en")
        else:
            self.ocr = None  
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=(config.device == 'cuda')
        )

        print("✓ PaddleOCR loaded")
    
    def extract_text(self, frames: List[Dict]) -> Dict:
        if not PADDLE_AVAILABLE:
            return {
                "text_timeline": {},
                "overlay_count": 0,
                "has_text": False
            }


        """
        Extract text from video frames
        
        Args:
            frames: List of frame dictionaries with 'frame' key
            
        Returns:
            Dictionary with text timeline and overlay count
        """
        
        text_timeline = {}
        total_text_instances = 0
        
        for frame_data in frames:
            timestamp = frame_data['timestamp']
            frame = frame_data['frame']
            
            try:
                # Run OCR
                result = self.ocr.ocr(frame, cls=True)
                
                if result and result[0]:
                    # Extract text
                    texts = []
                    for line in result[0]:
                        text = line[1][0]  # Get text content
                        confidence = line[1][1]  # Get confidence
                        
                        if confidence > 0.7:  # Confidence threshold
                            texts.append(text)
                            total_text_instances += 1
                    
                    if texts:
                        text_timeline[timestamp] = ' '.join(texts)
                
            except Exception as e:
                # OCR can fail on some frames
                continue
        
        return {
            'text_timeline': text_timeline,
            'overlay_count': total_text_instances,
            'has_text': total_text_instances > 0
        }
    
    def get_dominant_text(self, text_timeline: Dict) -> str:
        """Get most common text overlay"""
        
        if not text_timeline:
            return ""
        
        # Combine all text
        all_text = ' '.join(text_timeline.values())
        
        # Return first 100 chars
        return all_text[:100]




