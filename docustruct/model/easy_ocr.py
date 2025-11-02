"""
EasyOCR Model Implementation for Manazir OCR

Implements EasyOCR for quick and simple OCR tasks.
"""

from typing import Optional, List
from PIL import Image
import numpy as np
import torch

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class EasyOcrModel(BaseOcrModel):
    """EasyOCR model for simple and fast OCR"""
    
    def __init__(
        self,
        model_path: str = "easyocr",
        device: str = "cuda",
        languages: List[str] = None,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.languages = languages or ['ar', 'en']
        self.load_model()
    
    def load_model(self) -> None:
        """Load EasyOCR reader"""
        import logging

        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            )

        logger = logging.getLogger(__name__)
        logger.info(f"Loading EasyOCR with languages: {self.languages}...")

        gpu = self.device == "cuda" and torch.cuda.is_available()
        self.model = easyocr.Reader(self.languages, gpu=gpu)

        logger.info("EasyOCR loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with EasyOCR"""
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Perform OCR
        results = self.model.readtext(image_np)
        
        # Extract text and confidence
        texts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf)
        
        # Combine text
        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OcrResult(
            text=full_text,
            confidence=avg_confidence,
            model_name="EasyOCR",
            metadata={
                "languages": self.languages,
                "num_detections": len(results),
            }
        )
