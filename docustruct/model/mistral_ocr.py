"""
Mistral OCR Model Implementation for DocuStruct

Implements Mistral OCR API for enterprise-grade OCR.
"""

from typing import Optional
from PIL import Image
import base64
from io import BytesIO
import os

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class MistralOcrModel(BaseOcrModel):
    """Mistral OCR model for enterprise-grade OCR via API"""
    
    def __init__(
        self,
        model_path: str = "mistral-ocr",
        device: str = "cpu",  # API-based, no GPU needed
        api_key: Optional[str] = None,
        api_endpoint: str = "https://api.mistral.ai/v1/ocr",
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.api_endpoint = api_endpoint
        self.load_model()
    
    def load_model(self) -> None:
        """Initialize Mistral API client"""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is not installed. Install it with: pip install requests"
            )
        
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        print("Mistral OCR API client initialized successfully!")
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with Mistral OCR API"""
        
        # Encode image
        base64_image = self._encode_image(image)
        
        # Prepare request
        payload = {
            "image": base64_image,
            "format": "markdown"  # or "text", "json"
        }
        
        if prompt:
            payload["prompt"] = prompt
        
        # Call API
        response = self.session.post(self.api_endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text
        output_text = result.get("text", "")
        
        return OcrResult(
            text=output_text,
            confidence=result.get("confidence", 1.0),
            model_name="Mistral-OCR",
            metadata={
                "model": self.model_path,
                "format": "markdown",
            }
        )
