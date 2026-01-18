import cv2
from scenedetect import detect, ContentDetector
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

class VideoProcessor:
    """
    Video processing: metadata extraction, scene detection, frame sampling
    Uses OpenCV and PySceneDetect
    """
    
    def __init__(self, config):
        self.config = config
        self.cut_threshold = config.cfg['processing']['hard_cut_threshold']
        self.frame_interval = config.cfg['processing']['frame_sampling_interval']
        self.max_frames = config.cfg['processing']['max_frames_per_video']
    
    def extract_metadata(self, video_path: str) -> Dict:
        """Extract video metadata using OpenCV"""
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            'path': video_path,
            'filename': Path(video_path).name,
            'duration': round(duration, 2),
            'fps': round(fps, 2),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': frame_count,
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        return metadata
    
    def detect_hard_cuts(self, video_path: str) -> List[float]:
        """
        Detect hard cuts using PySceneDetect
        Returns list of cut timestamps in seconds
        """
        
        try:
            scene_list = detect(
                video_path,
                ContentDetector(threshold=self.cut_threshold)
            )
            
            # Extract cut timestamps (start of each scene)
            cut_times = [scene[0].get_seconds() for scene in scene_list]
            
            return cut_times
            
        except Exception as e:
            print(f"  Warning: Scene detection failed: {e}")
            return []
    
    def extract_frames(self, video_path: str, cut_times: List[float]) -> List[Dict]:
        """
        Extract frames at scene cuts + regular intervals
        
        Returns:
            List of frame dictionaries with timestamp and image data
        """
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        # Build sampling times
        sample_times = set(cut_times)
        
        # Add regular interval samples
        num_intervals = int(duration / self.frame_interval)
        for i in range(num_intervals + 1):
            t = i * self.frame_interval
            if t <= duration:
                sample_times.add(t)
        
        # Limit total frames
        sample_times = sorted(list(sample_times))[:self.max_frames]
        
        frames = []
        
        for timestamp in sample_times:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append({
                    'timestamp': round(timestamp, 2),
                    'frame_number': frame_number,
                    'frame': frame_rgb,
                    'shape': frame_rgb.shape
                })
        
        cap.release()
        
        return frames