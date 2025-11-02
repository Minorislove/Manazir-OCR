"""
Surya OCR Model Implementation for DocuStruct

Implements Surya OCR for comprehensive document understanding.
"""

from typing import Optional, List
from PIL import Image

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class SuryaOcrModel(BaseOcrModel):
    """Surya OCR model for comprehensive document understanding"""
    
    def __init__(
        self,
        model_path: str = "hesham-haroon/surya",
        device: str = "cuda",
        languages: List[str] = None,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.languages = languages or ['ar', 'en']
        self.load_model()
    
    def load_model(self) -> None:
        """Load Surya OCR models"""
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
        except ImportError:
            raise ImportError(
                "Surya OCR is not installed. Install it with: pip install surya-ocr"
            )
        
        print(f"Loading Surya OCR models...")
        
        # Load detection model
        self.det_model = load_det_model()
        self.det_processor = load_det_processor()
        
        # Load recognition model
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        
        # Store the run_ocr function
        self.run_ocr = run_ocr
        
        print("Surya OCR models loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with Surya OCR"""
        
        # Run OCR
        predictions = self.run_ocr(
            [image],
            [self.languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor
        )
        
        # Extract text from predictions
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            texts = []
            confidences = []
            
            for text_line in pred.text_lines:
                texts.append(text_line.text)
                if hasattr(text_line, 'confidence'):
                    confidences.append(text_line.confidence)
            
            full_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        else:
            full_text = ""
            avg_confidence = 0.0
        
        return OcrResult(
            text=full_text,
            confidence=avg_confidence,
            model_name="Surya-OCR",
            metadata={
                "languages": self.languages,
                "num_lines": len(texts) if predictions else 0,
            }
        )
