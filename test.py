"""
LOCAL-SAFE PRODUCTION TEST (CPU LAPTOP FRIENDLY)

Uses ALL src/ modules exactly like main.py,
but SKIPS Qwen when no GPU is available.

File structure expected:

Analyser_vid/
├── data/
│   ├── taxonomy/refined_lists.csv
│   └── videos/  (at least one .mp4 here)
├── src/
│   ├── config.py
│   ├── video_processor.py
│   ├── audio_analyzer.py
│   ├── qwen_analyzer.py
│   ├── attractor_classifier.py
│   ├── narrative_classifier.py
│   ├── aggregator.py
│   ├── validator.py
│   └── logger.py
├── main.py        (your production file)
└── test_local.py  (THIS FILE)
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import torch

# Add src to path (critical)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import your real modules
from config import Config
from video_processor import VideoProcessor
from audio_analyzer import AudioAnalyzer
from qwen_analyzer import QwenAnalyzer
from attractor_classifier import AttractorClassifier
from narrative_classifier import NarrativeClassifier
from aggregator import ResultAggregator
from validator import ResultValidator
from logger import setup_logger


class LocalSafeConfig:
    """
    Minimal config that matches what your modules expect,
    but works on a CPU laptop.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

        # Core paths
        self.video_dir = str(base_dir / "data" / "videos")
        self.output_dir = str(base_dir / "output")
        self.log_file = str(base_dir / "output" / "test_local_log.txt")

        # Device auto-detect
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output folder
        os.makedirs(self.output_dir, exist_ok=True)

        # Mimic your real config structure
        self.cfg = {
            "processing": {
                "batch_size": 8,
                "frame_sampling_interval": 2,
                "hard_cut_threshold": 30,
                "max_frames_per_video": 50,
            },
            "models": {
                "qwen_model": "Qwen/Qwen2-VL-7B-Instruct",
                "whisper_model": "base",
                "device": self.device,
            },
            "validation": {
                "strict_mode": False,
                "auto_normalize": True,
                "confidence_threshold": 0.6,
            },
            "output": {
                "log_file": self.log_file,
                "save_frames": False,
                "incremental_save": True,
            },
        }


class ProductionVideoTester:
    def __init__(self):
        print("\n" + "=" * 80)
        print("LOCAL SAFE PRODUCTION VIDEO TEST")
        print("Uses ALL src/ modules — CPU safe")
        print("=" * 80 + "\n")

        base_dir = Path(__file__).parent
        self.config = LocalSafeConfig(base_dir)

        # Logger
        self.logger = setup_logger("test_local", self.config.log_file)
        self.logger.info("Local production test started")

        print(f"Running on device: {self.config.device}")

        # Initialize modules (same as main.py)
        print("\nInitializing modules...\n")

        self.video_processor = VideoProcessor(self.config)
        self.audio_analyzer = AudioAnalyzer(self.config)

        # --- KEY PART: SKIP QWEN LOCALLY IF NO GPU ---
        if self.config.device == "cpu":
            print("⚠️ SKIPPING Qwen locally (CPU detected)")
            self.qwen_analyzer = None
        else:
            print("Loading Qwen (GPU available)...")
            self.qwen_analyzer = QwenAnalyzer(self.config)

        self.attractor_classifier = AttractorClassifier(self.config)
        self.narrative_classifier = NarrativeClassifier(self.config)
        self.aggregator = ResultAggregator(self.config)
        self.validator = ResultValidator(self.config)

        print("\n✓ All modules initialized (Qwen skipped on CPU)\n")

    def find_test_video(self):
        print(f"Searching in: {self.config.video_dir}")

        if not os.path.exists(self.config.video_dir):
            raise FileNotFoundError("data/videos folder missing!")

        videos = [
            f for f in os.listdir(self.config.video_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]

        if not videos:
            raise FileNotFoundError("No videos found in data/videos")

        video_name = videos[0]
        video_path = os.path.join(self.config.video_dir, video_name)

        print(f"✓ Using video: {video_name}\n")
        return video_path, video_name

    async def analyze_video(self, video_path, video_id):
        print("=" * 80)
        print(f"ANALYZING: {video_id}")
        print("=" * 80 + "\n")

        try:
            # ---- STEP 1: TECHNICAL METRICS ----
            technical = self.video_processor.extract_metadata(video_path)
            print(f"Duration: {technical['duration']:.1f}s")

            # ---- STEP 2: HARD CUTS ----
            cuts = self.video_processor.detect_hard_cuts(video_path)
            technical["total_cuts"] = len(cuts)
            technical["cuts_per_minute"] = (
                len(cuts) / max(1.0, technical["duration"]) * 60
            )
            print(f"Hard cuts: {len(cuts)}")

            # ---- STEP 3: FRAMES ----
            frames = self.video_processor.extract_frames(video_path, cuts)
            print(f"Frames extracted: {len(frames)}")

            # ---- STEP 4: AUDIO (WHISPER) ----
            audio_result = self.audio_analyzer.analyze(video_path)
            technical["audio_type"] = audio_result.get("type", "unknown")
            print(f"Audio type: {technical['audio_type']}")

            # ---- STEP 5: OCR (stub for local) ----
            ocr_result = {"text_timeline": {}, "overlay_count": 0}
            technical["text_overlay_count"] = 0

            # ---- STEP 6: QWEN (only if GPU) ----
            if self.qwen_analyzer:
                print("Running Qwen analysis...")
                qwen_result = await self.qwen_analyzer.analyze_video(
                    frames=frames,
                    audio_transcript=audio_result.get("transcript", ""),
                    ocr_text=ocr_result["text_timeline"],
                    duration=technical["duration"],
                )
            else:
                print("Skipping Qwen — using placeholder result")
                qwen_result = {
                    "segments": [],
                    "confidence": 0.0,
                }

            # ---- STEP 7: CLASSIFICATION ----
            attractor_result = self.attractor_classifier.classify(
                qwen_result, technical["duration"]
            )

            narrative_result = self.narrative_classifier.classify(
                qwen_result,
                audio_result,
                ocr_result,
                technical["duration"],
            )

            # ---- AGGREGATION ----
            complete_result = self.aggregator.aggregate(
                video_id=video_id,
                video_path=video_path,
                technical=technical,
                attractors=attractor_result,
                narratives=narrative_result,
            )

            # ---- VALIDATION ----
            is_valid, errors = self.validator.validate(
                complete_result, technical["duration"]
            )

            if not is_valid:
                print("⚠️ Normalizing results...")
                complete_result = self.validator.normalize(
                    complete_result, technical["duration"]
                )

            print("\n✓ Local analysis completed!\n")
            return complete_result

        except Exception as e:
            self.logger.error("Test failed", exc_info=True)
            return {"video_id": video_id, "status": "failed", "error": str(e)}

    def save_results(self, result, video_id):
        import json
        from datetime import datetime

        def make_json_safe(obj):
            """Recursively convert non-JSON types into safe formats."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(x) for x in obj]
            else:
                return obj

        safe_result = make_json_safe(result)

        out = os.path.join(
            self.config.output_dir, f"test_local_{video_id}.json"
        )

        with open(out, "w", encoding="utf-8") as f:
            json.dump(safe_result, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {out}\n")
        return out


async def main():
    tester = ProductionVideoTester()

    video_path, video_name = tester.find_test_video()
    video_id = Path(video_name).stem

    result = await tester.analyze_video(video_path, video_id)

    out_file = tester.save_results(result, video_id)

    print("=" * 80)
    print("LOCAL TEST COMPLETE")
    print("=" * 80)
    print(f"Results in: {out_file}\n")


if __name__ == "__main__":
    asyncio.run(main())
