"""
OCR-RL2 Model Implementation for Manazir OCR

Implements the OCR-RL2 model for Arabic OCR.
Model by Mohamed Zayton: https://huggingface.co/Mohamed-Zayton/ocr_rl2
"""

from typing import Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import logging

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class OcrRl2Model(BaseOcrModel):
    """OCR-RL2 model for Arabic OCR"""
    
    def __init__(
        self,
        model_path: str = "Mohamed-Zayton/ocr_rl2",
        device: str = "cuda",
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.max_tokens = max_tokens
        self.load_model()
    
    def load_model(self) -> None:
        """Load OCR-RL2 model and processor"""
        logger = logging.getLogger(__name__)
        logger.info(f"Loading OCR-RL2 model from {self.model_path}...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            logger.info("OCR-RL2 model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading OCR-RL2 model: {e}")
            logger.info("Falling back to Qwen2-VL architecture...")
            
            # Fallback to Qwen2-VL if model has compatibility issues
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logger.info("Loaded with Qwen2-VL architecture!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with OCR-RL2 model"""
        
        if prompt is None:
            prompt = "استخرج النص من هذه الصورة. Extract all text from this image."
        
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
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except:
            # Fallback if qwen_vl_utils is not available
            image_inputs = [image]
            video_inputs = None
        
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
            confidence=1.0,
            model_name="OCR-RL2",
            metadata={
                "model_path": self.model_path,
                "prompt": prompt,
                "language": "ar",
            }
        )
