"""
COMPLETE MULTIMODAL VIDEO ANALYSIS SYSTEM
Production-Ready Implementation for 284 Tourism Videos
Author: Professional AI Research System
Version: 1.0
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

# Import all modules
from src.config import Config
from src.drive_downloader import DriveDownloader
from src.video_processor import VideoProcessor
from src.audio_analyzer import AudioAnalyzer
from src.ocr_detector import OCRDetector
from src.qwen_analyzer import QwenAnalyzer
from src.attractor_classifier import AttractorClassifier
from src.narrative_classifier import NarrativeClassifier
from src.aggregator import ResultAggregator
from src.validator import ResultValidator
from src.excel_writer import ExcelWriter
from src.logger import setup_logger

class VideoAnalysisOrchestrator:
    """
    Main orchestrator for automated video analysis system
    Coordinates all processing steps for 284 videos
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the system with configuration"""
        self.config = Config(config_path)
        self.logger = setup_logger("main", self.config.log_file)
        
        # Initialize all components
        self.logger.info("Initializing Video Analysis System...")
        
        self.drive_downloader = DriveDownloader(self.config)
        self.video_processor = VideoProcessor(self.config)
        self.audio_analyzer = AudioAnalyzer(self.config)
        self.ocr_detector = OCRDetector(self.config)
        self.qwen_analyzer = QwenAnalyzer(self.config)
        self.attractor_classifier = AttractorClassifier(self.config)
        self.narrative_classifier = NarrativeClassifier(self.config)
        self.aggregator = ResultAggregator(self.config)
        self.validator = ResultValidator(self.config)
        self.excel_writer = ExcelWriter(self.config)
        
        self.logger.info("✓ All components initialized successfully")
        
    async def setup(self):
        """Setup: Download videos and load taxonomy"""
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1: SETUP & PREPARATION")
        self.logger.info("=" * 80)
        
        # Step 1: Download videos from Google Drive
        if self.config.download_videos:
            self.logger.info("Downloading videos from Google Drive...")
            await self.drive_downloader.download_all_videos()
            self.logger.info("✓ Videos downloaded")
        else:
            self.logger.info("Using existing local videos")
        
        # Step 2: Load Excel and taxonomy
        self.logger.info("Loading Excel workbook...")
        self.excel_writer.load_workbook()
        self.logger.info("✓ Excel loaded")
        
        # Step 3: Load taxonomy from Refined_Lists sheet
        self.logger.info("Loading taxonomy...")
        taxonomy = self.excel_writer.load_taxonomy()
        self.attractor_classifier.load_taxonomy(taxonomy)
        self.narrative_classifier.load_taxonomy(taxonomy)
        self.logger.info("✓ Taxonomy loaded: 19 attractors, 4 frames")
        
    async def process_single_video(self, video_path: str, video_id: str) -> Dict:
        """
        Process a single video through complete pipeline
        
        Args:
            video_path: Path to video file
            video_id: Unique video identifier (e.g., Tourist_001)
            
        Returns:
            Complete analysis results dictionary
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing: {video_id} - {Path(video_path).name}")
        self.logger.info(f"{'='*80}")
        
        try:
            # STEP 1: Extract technical metrics
            self.logger.info("  [1/7] Extracting technical metrics...")
            technical = self.video_processor.extract_metadata(video_path)
            self.logger.info(f"    Duration: {technical['duration']}s | FPS: {technical['fps']}")
            
            # STEP 2: Scene detection (hard cuts only)
            self.logger.info("  [2/7] Detecting hard cuts...")
            cuts = self.video_processor.detect_hard_cuts(video_path)
            technical['total_cuts'] = len(cuts)
            technical['cuts_per_minute'] = (len(cuts) / technical['duration']) * 60
            self.logger.info(f"    Hard cuts: {len(cuts)} ({technical['cuts_per_minute']:.1f}/min)")
            
            # STEP 3: Frame extraction
            self.logger.info("  [3/7] Extracting keyframes...")
            frames = self.video_processor.extract_frames(video_path, cuts)
            self.logger.info(f"    Extracted {len(frames)} frames for analysis")
            
            # STEP 4: Audio analysis
            self.logger.info("  [4/7] Analyzing audio...")
            audio_result = self.audio_analyzer.analyze(video_path)
            technical['audio_type'] = audio_result['type']
            self.logger.info(f"    Audio type: {audio_result['type']}")
            
            # STEP 5: OCR text extraction
            self.logger.info("  [5/7] Extracting on-screen text...")
            ocr_result = self.ocr_detector.extract_text(frames)
            technical['text_overlay_count'] = ocr_result['overlay_count']
            self.logger.info(f"    Text overlays: {ocr_result['overlay_count']}")
            
            # STEP 6: Qwen-VL multimodal analysis
            self.logger.info("  [6/7] Running Qwen-VL multimodal analysis...")
            qwen_result = await self.qwen_analyzer.analyze_video(
                frames=frames,
                audio_transcript=audio_result['transcript'],
                ocr_text=ocr_result['text_timeline'],
                duration=technical['duration']
            )
            self.logger.info(f"    Qwen analysis complete")
            
            # STEP 7: Classification & aggregation
            self.logger.info("  [7/7] Classifying attractors & narrative frames...")
            
            # Classify attractors (19 categories)
            attractor_result = self.attractor_classifier.classify(
                qwen_result,
                technical['duration']
            )
            
            # Classify narrative frames (AE/LI/IN/CR × 3 modalities)
            narrative_result = self.narrative_classifier.classify(
                qwen_result,
                audio_result,
                ocr_result,
                technical['duration']
            )
            
            # STEP 8: Aggregate & validate
            complete_result = self.aggregator.aggregate(
                video_id=video_id,
                video_path=video_path,
                technical=technical,
                attractors=attractor_result,
                narratives=narrative_result
            )
            
            # Validate results
            is_valid, errors = self.validator.validate(complete_result, technical['duration'])
            
            if not is_valid:
                self.logger.warning(f"    Validation warnings: {errors}")
                # Auto-normalize if needed
                complete_result = self.validator.normalize(complete_result, technical['duration'])
                self.logger.info(f"    ✓ Results normalized")
            
            self.logger.info(f"  ✓ {video_id} processing complete")
            
            return complete_result
            
        except Exception as e:
            self.logger.error(f"  ✗ Error processing {video_id}: {str(e)}", exc_info=True)
            return {
                'video_id': video_id,
                'error': str(e),
                'status': 'failed'
            }
    
    async def process_all_videos(self):
        """Process all 284 videos in batch"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 2: BATCH VIDEO PROCESSING")
        self.logger.info("=" * 80)
        
        # Get list of videos to process from Excel
        video_list = self.excel_writer.get_video_list()
        total_videos = len(video_list)
        
        self.logger.info(f"\nTotal videos to process: {total_videos}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = []
        failed_count = 0
        
        for idx, video_info in enumerate(video_list, 1):
            video_id = video_info['video_id']
            video_filename = video_info['video_file_name']
            video_path = os.path.join(self.config.video_dir, video_filename)
            
            # Progress indicator
            self.logger.info(f"\n[{idx}/{total_videos}] Starting {video_id}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                self.logger.error(f"  ✗ Video file not found: {video_path}")
                failed_count += 1
                continue
            
            # Process video
            result = await self.process_single_video(video_path, video_id)
            
            if result.get('status') == 'failed':
                failed_count += 1
            else:
                # Write to Excel immediately (incremental save)
                self.excel_writer.write_video_result(result)
                self.logger.info(f"  ✓ Written to Excel")
            
            results.append(result)
            
            # Progress summary every 10 videos
            if idx % 10 == 0:
                success_count = idx - failed_count
                self.logger.info(f"\n--- Progress: {idx}/{total_videos} ---")
                self.logger.info(f"    Successful: {success_count}")
                self.logger.info(f"    Failed: {failed_count}")
                self.logger.info(f"    Success rate: {(success_count/idx)*100:.1f}%")
        
        return results
    
    async def finalize(self, results: List[Dict]):
        """Finalize processing and generate reports"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 3: FINALIZATION")
        self.logger.info("=" * 80)
        
        # Save Excel file
        self.logger.info("Saving Excel workbook...")
        self.excel_writer.save_workbook()
        self.logger.info(f"✓ Excel saved: {self.config.excel_output_path}")
        
        # Generate summary report
        self.logger.info("\nGenerating summary report...")
        self._generate_summary_report(results)
        
        # Generate methodology document
        self.logger.info("Generating methodology document...")
        self._generate_methodology()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ALL PROCESSING COMPLETE")
        self.logger.info("=" * 80)
        
    def _generate_summary_report(self, results: List[Dict]):
        """Generate processing summary report"""
        total = len(results)
        successful = len([r for r in results if r.get('status') != 'failed'])
        failed = total - successful
        
        report = f"""
PROCESSING SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEOS PROCESSED
================
Total videos: {total}
Successful: {successful}
Failed: {failed}
Success rate: {(successful/total)*100:.1f}%

OUTPUT FILES
============
Excel file: {self.config.excel_output_path}
Logs: {self.config.log_file}
"""
        
        report_path = os.path.join(self.config.output_dir, "processing_summary.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"✓ Summary report: {report_path}")
        
    def _generate_methodology(self):
        """Generate academic methodology document"""
        methodology = """
METHODOLOGY DOCUMENT
Automated Multimodal Video Analysis for Destination Branding Research

1. TECHNICAL PIPELINE
   - Video ingestion via FFmpeg
   - Scene detection using PySceneDetect (threshold=30)
   - Frame sampling: cuts + 2-second intervals
   - Audio transcription: OpenAI Whisper (base model)
   - OCR: PaddleOCR
   - Multimodal analysis: Qwen-2.5-VL-7B-Instruct

2. DESTINATION ATTRACTOR CLASSIFICATION
   - 19 predefined categories (Vinyals-Mirabent, 2019)
   - Second-by-second temporal attribution
   - Dominance scoring algorithm
   - Absorption principle enforcement

3. NARRATIVE FRAME CLASSIFICATION
   - Four frames: AE/LI/IN/CR (Sallaku, 2025)
   - Three modalities: Visual/Audio/Text
   - Independent classification per modality
   - Tag assignment from controlled vocabulary

4. VALIDATION & NORMALIZATION
   - Constraint: Σ(attractor_seconds) = video_duration
   - Constraint: Σ(frame_seconds_per_modality) = video_duration
   - Proportional normalization when needed

5. REPRODUCIBILITY
   - Fixed random seeds
   - Deterministic processing
   - Complete logging of all decisions
"""
        
        method_path = os.path.join(self.config.output_dir, "methodology.txt")
        with open(method_path, 'w') as f:
            f.write(methodology)
        
        self.logger.info(f"✓ Methodology: {method_path}")

async def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MULTIMODAL VIDEO ANALYSIS SYSTEM                           ║
    ║  Automated Processing for 284 Tourism Videos                ║
    ║  Version 1.0 - Production Build                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize system
        orchestrator = VideoAnalysisOrchestrator()
        
        # Phase 1: Setup
        await orchestrator.setup()
        
        # Phase 2: Process all videos
        results = await orchestrator.process_all_videos()
        
        # Phase 3: Finalize
        await orchestrator.finalize(results)
        
        print("\n✓ SUCCESS: All processing complete!")
        print(f"Check logs: {orchestrator.config.log_file}")
        
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())