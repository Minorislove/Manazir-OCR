"""
OpenAI GPT-4o Model Implementation for Manazir OCR

Implements OpenAI GPT-4o for OCR via API.
"""

from typing import Optional
from PIL import Image
import base64
from io import BytesIO
import os
import logging

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class OpenAIGPT4OModel(BaseOcrModel):
    """OpenAI GPT-4o model for OCR via API"""
    
    def __init__(
        self,
        model_path: str = "gpt-4o",
        device: str = "cpu",  # API-based, no GPU needed
        api_key: Optional[str] = None,
        api_endpoint: str = "https://api.openai.com/v1/chat/completions",
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_endpoint = api_endpoint
        self.max_tokens = max_tokens
        self.load_model()
    
    def load_model(self) -> None:
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI SDK is not installed. Install it with: pip install openai"
            )
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Derive base_url from the provided api_endpoint (expecting .../v1/...)
        base_url = None
        try:
            if "/v1" in self.api_endpoint:
                base_url = self.api_endpoint.split("/v1")[0] + "/v1"
        except Exception:
            base_url = None
        if base_url:
            self.model = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.model = OpenAI(api_key=self.api_key)
        logger = logging.getLogger(__name__)
        logger.info("OpenAI GPT-4o client initialized successfully!")
    
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
        """Process image with OpenAI GPT-4o"""
        
        if prompt is None:
            prompt = "Extract all text from this image. Preserve the layout and structure. Return only the text content."
        
        # Encode image
        base64_image = self._encode_image(image)
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call API
        response = self.model.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=self.max_tokens
        )
        
        # Extract text
        output_text = response.choices[0].message.content
        
        return OcrResult(
            text=output_text,
            confidence=1.0,  # API doesn't provide confidence
            model_name="GPT-4o",
            metadata={
                "model": self.model_path,
                "prompt": prompt,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        )
