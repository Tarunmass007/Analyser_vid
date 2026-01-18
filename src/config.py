import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """
    Configuration management for video analysis system
    Loads from config.yaml or provides sensible defaults
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = self._default_config()
            self._save_default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if no config.yaml exists"""
        return {
            'google_drive': {
                'folder_id': '1IybH4QIhUO1sCoijKiYC8GTjbTKpeyuK',
                'download_videos': False,
                'output_dir': 'data/videos'
            },
            'excel': {
                'input_path': 'data/Video_Coding_Sheet.xlsx',
                'output_path': 'output/coded_videos.xlsx',
                'main_sheet': 'Coding_TouristUCG',
                'taxonomy_sheet': 'Refined_Lists'
            },
            'processing': {
                'batch_size': 8,
                'frame_sampling_interval': 2.0,
                'hard_cut_threshold': 30,
                'max_frames_per_video': 100,
                'parallel_workers': 1
            },
            'models': {
                'qwen_model': 'Qwen/Qwen2-VL-7B-Instruct',
                'whisper_model': 'base',
                'device': 'cuda',
                'use_fp16': True
            },
            'output': {
                'log_file': 'output/processing_log.txt',
                'save_frames': False,
                'incremental_save': True,
                'checkpoint_interval': 10
            },
            'validation': {
                'strict_mode': False,
                'auto_normalize': True,
                'confidence_threshold': 0.6,
                'min_attractor_seconds': 0.5
            }
        }
    
    def _save_default_config(self):
        """Save default config to file"""
        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
    
    # Convenient property accessors
    @property
    def drive_folder_id(self) -> str:
        return self.cfg['google_drive']['folder_id']
    
    @property
    def download_videos(self) -> bool:
        return self.cfg['google_drive']['download_videos']
    
    @property
    def video_dir(self) -> str:
        return self.cfg['google_drive']['output_dir']
    
    @property
    def excel_path(self) -> str:
        return self.cfg['excel']['input_path']
    
    @property
    def excel_output_path(self) -> str:
        return self.cfg['excel']['output_path']
    
    @property
    def main_sheet_name(self) -> str:
        return self.cfg['excel']['main_sheet']
    
    @property
    def taxonomy_sheet_name(self) -> str:
        return self.cfg['excel']['taxonomy_sheet']
    
    @property
    def log_file(self) -> str:
        return self.cfg['output']['log_file']
    
    @property
    def output_dir(self) -> Path:
        return Path(self.cfg['output']['log_file']).parent
    
    @property
    def device(self) -> str:
        return self.cfg['models']['device']
