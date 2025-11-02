"""
Qari-OCR Model Implementation for Manazir OCR

Implements the Qari-OCR model specialized for Arabic text recognition.
"""

from typing import Optional
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class QariOcrModel(BaseOcrModel):
    """Qari-OCR model for high-accuracy Arabic text recognition"""
    
    def __init__(
        self,
        model_path: str = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
        device: str = "cuda",
        max_tokens: int = 2000,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.max_tokens = max_tokens
        self.load_model()
    
    def load_model(self) -> None:
        """Load Qari-OCR model and processor"""
        logger = logging.getLogger(__name__)
        logger.info(f"Loading Qari-OCR model from {self.model_path}...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        logger.info("Qari-OCR model loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with Qari-OCR model"""
        
        if prompt is None:
            prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally. Do not hallucinate."
        
        # Prepare messages with in-memory image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens
        )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return OcrResult(
            text=output_text,
            confidence=1.0,  # Qari-OCR doesn't provide confidence scores
            model_name="Qari-OCR-0.1",
            metadata={
                "model_path": self.model_path,
                "prompt": prompt,
                "language": "ar",
            }
        )
