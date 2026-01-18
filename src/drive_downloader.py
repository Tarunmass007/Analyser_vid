import gdown
import os
from pathlib import Path
from typing import List

class DriveDownloader:
    """
    Download videos from Google Drive folder
    Uses gdown library for public folder access
    """
    
    def __init__(self, config):
        self.config = config
        self.folder_id = config.drive_folder_id
        self.output_dir = Path(config.video_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_all_videos(self):
        """Download entire folder from Google Drive"""
        
        print(f"\n{'='*80}")
        print(f"DOWNLOADING VIDEOS FROM GOOGLE DRIVE")
        print(f"{'='*80}")
        print(f"Folder ID: {self.folder_id}")
        print(f"Output: {self.output_dir}")
        
        folder_url = f"https://drive.google.com/drive/folders/{self.folder_id}"
        
        try:
            # Download folder recursively
            gdown.download_folder(
                url=folder_url,
                output=str(self.output_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True
            )
            
            # Count downloaded files
            video_files = list(self.output_dir.glob('*.mp4')) + \
                         list(self.output_dir.glob('*.MP4')) + \
                         list(self.output_dir.glob('*.mov'))
            
            print(f"\n✓ Downloaded {len(video_files)} video files")
            
        except Exception as e:
            print(f"\n⚠ Google Drive download error: {e}")
            print(f"Please manually download videos to: {self.output_dir}")
            print("Then set download_videos: false in config.yaml")
    
    def verify_downloads(self, expected_count: int = 284) -> bool:
        """Verify all videos are downloaded"""
        
        video_files = list(self.output_dir.glob('*.mp4')) + \
                     list(self.output_dir.glob('*.MP4'))
        
        actual_count = len(video_files)
        
        if actual_count >= expected_count:
            print(f"✓ All {actual_count} videos present")
            return True
        else:
            print(f"⚠ Only {actual_count}/{expected_count} videos found")
            return False