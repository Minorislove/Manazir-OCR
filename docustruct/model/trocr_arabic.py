"""
TrOCR Arabic Model Implementation for DocuStruct

Implements TrOCR for Arabic handwritten text recognition.
"""

from typing import Optional
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class TrOcrArabicModel(BaseOcrModel):
    """TrOCR model for Arabic handwritten text recognition"""
    
    def __init__(
        self,
        model_path: str = "David-Magdy/TR_OCR_LARGE",
        device: str = "cuda",
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.max_length = max_length
        self.load_model()
    
    def load_model(self) -> None:
        """Load TrOCR model and processor"""
        print(f"Loading TrOCR Arabic model from {self.model_path}...")
        
        self.processor = TrOCRProcessor.from_pretrained(self.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("TrOCR Arabic model loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with TrOCR Arabic model"""
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=self.max_length
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return OcrResult(
            text=generated_text,
            confidence=1.0,  # TrOCR doesn't provide confidence scores
            model_name="TrOCR-Arabic",
            metadata={
                "model_path": self.model_path,
                "language": "ar",
                "type": "handwritten",
            }
        )
