"""
Tesseract OCR Model Implementation for Manazir OCR

Implements Tesseract OCR for baseline text recognition.
"""

from typing import Optional
from PIL import Image
import logging

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class TesseractModel(BaseOcrModel):
    """Tesseract OCR model for baseline text recognition"""
    
    def __init__(
        self,
        model_path: str = "tesseract",
        device: str = "cpu",  # Tesseract runs on CPU
        language: str = "ara+eng",
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.language = language
        self.load_model()
    
    def load_model(self) -> None:
        """Load Tesseract OCR"""
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            )
        
        self.model = pytesseract
        logger = logging.getLogger(__name__)
        logger.info(f"Tesseract OCR loaded with language: {self.language}")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with Tesseract OCR"""
        
        # Extract text
        text = self.model.image_to_string(image, lang=self.language)
        
        # Get confidence data
        try:
            data = self.model.image_to_data(image, lang=self.language, output_type=self.model.Output.DICT)
            confidences = [float(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        except:
            avg_confidence = 0.0
        
        return OcrResult(
            text=text.strip(),
            confidence=avg_confidence / 100.0,  # Normalize to 0-1
            model_name="Tesseract",
            metadata={
                "language": self.language,
            }
        )
