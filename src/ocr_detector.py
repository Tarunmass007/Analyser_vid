# ======== FIXED OCRDetector (COLAB + RUNPOD SAFE) ========

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    print("⚠️ PaddleOCR disabled in this environment:", e)
    PADDLE_AVAILABLE = False


class OCRDetector:
    def __init__(self, config):
        self.config = config

        if not PADDLE_AVAILABLE:
            print("⚠️ OCR disabled: PaddleOCR not usable here.")
            self.ocr = None
            return

        print("Initializing PaddleOCR (auto GPU if available)...")

        # 👉 IMPORTANT: DO NOT PASS show_log or use_gpu
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en"
        )

        print("✓ PaddleOCR ready")

    def extract_text(self, frame):
        if not PADDLE_AVAILABLE or self.ocr is None:
            return {
                "text_timeline": {},
                "overlay_count": 0,
                "has_text": False
            }

        try:
            result = self.ocr.ocr(frame, cls=True)

            if not result or len(result) == 0:
                return {
                    "text_timeline": {},
                    "overlay_count": 0,
                    "has_text": False
                }

            texts = []
            for line in result:
                if line:
                    for item in line:
                        text = item[1][0]
                        confidence = item[1][1]
                        texts.append({"text": text, "confidence": confidence})

            return {
                "text_timeline": {"0-1": texts},
                "overlay_count": len(texts),
                "has_text": len(texts) > 0
            }

        except Exception as e:
            print("⚠️ OCR failed on frame:", e)
            return {
                "text_timeline": {},
                "overlay_count": 0,
                "has_text": False
            }
