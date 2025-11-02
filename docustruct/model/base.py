"""
Base Model Interface for DocuStruct OCR Framework

This module defines the abstract base class for all OCR models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image

from docustruct.model.schema import OcrResult


class BaseOcrModel(ABC):
    """Abstract base class for all OCR models"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize the OCR model
        
        Args:
            model_path: Path or identifier for the model
            device: Device to run the model on ('cuda' or 'cpu')
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.config = kwargs
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OcrResult:
        """
        Process a single image and return OCR result
        
        Args:
            image: PIL Image object
            prompt: Optional prompt for the model
            **kwargs: Additional processing parameters
        
        Returns:
            OcrResult object containing the extracted text and metadata
        """
        pass
    
    def process_images(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[OcrResult]:
        """
        Process multiple images
        
        Args:
            images: List of PIL Image objects
            prompts: Optional list of prompts for each image
            **kwargs: Additional processing parameters
        
        Returns:
            List of OcrResult objects
        """
        if prompts is None:
            prompts = [None] * len(images)
        
        results = []
        for image, prompt in zip(images, prompts):
            result = self.process_image(image, prompt, **kwargs)
            results.append(result)
        
        return results
    
    def process_pdf(
        self,
        pdf_path: Path,
        **kwargs
    ) -> List[OcrResult]:
        """
        Process a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            **kwargs: Additional processing parameters
        
        Returns:
            List of OcrResult objects, one per page
        """
        from docustruct.input import load_pdf_images
        
        images = load_pdf_images(pdf_path)
        return self.process_images(images, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "config": self.config,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path}, device={self.device})"
