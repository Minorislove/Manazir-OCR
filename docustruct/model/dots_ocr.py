"""
dots.ocr Model Implementation for DocuStruct

Implements the dots.ocr multilingual document parser.
"""

from typing import Optional
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

from docustruct.model.base import BaseOcrModel
from docustruct.model.schema import OcrResult


class DotsOcrModel(BaseOcrModel):
    """dots.ocr model for multilingual document parsing"""
    
    def __init__(
        self,
        model_path: str = "rednote-hilab/dots.ocr",
        device: str = "cuda",
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.max_tokens = max_tokens
        self.load_model()
    
    def load_model(self) -> None:
        """Load dots.ocr model and processor"""
        print(f"Loading dots.ocr model from {self.model_path}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print("dots.ocr model loaded successfully!")
    
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """Process image with dots.ocr model"""
        
        if prompt is None:
            prompt = "Extract all text from this document, preserving layout and structure."
        
        # Prepare messages
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
            confidence=1.0,
            model_name="dots.ocr",
            metadata={
                "model_path": self.model_path,
                "prompt": prompt,
            }
        )
