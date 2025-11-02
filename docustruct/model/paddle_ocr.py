"""
PaddleOCR Model Implementation for DocuStruct

Implements PaddleOCR for lightweight Arabic OCR.
"""

from typing import Optional
from PIL import Image
import numpy as np

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class PaddleOcrArabicModel(BaseOcrModel):
    """PaddleOCR model for lightweight Arabic text recognition"""
    
    def __init__(
        self,
        model_path: str = "PaddlePaddle/arabic_PP-OCRv3_mobile_rec",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.load_model()
    
    def load_model(self) -> None:
        """Load PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr"
            )
        
        print(f"Loading PaddleOCR Arabic model...")
        
        # Initialize PaddleOCR with Arabic support
        use_gpu = self.device == "cuda"
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang='ar',
            use_gpu=use_gpu,
            show_log=False
        )
        
        print("PaddleOCR Arabic model loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with PaddleOCR"""
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Perform OCR
        results = self.model.ocr(image_np, cls=True)
        
        # Extract text and confidence
        texts = []
        confidences = []
        
        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, conf = text_info[0], text_info[1]
                        texts.append(text)
                        confidences.append(conf)
        
        # Combine text
        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OcrResult(
            text=full_text,
            confidence=avg_confidence,
            model_name="PaddleOCR-Arabic",
            metadata={
                "language": "ar",
                "num_lines": len(texts),
            }
        )
