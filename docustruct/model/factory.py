"""
Model Factory for DocuStruct OCR Framework

This module provides a factory for creating OCR model instances.
"""

from typing import Optional, Dict, Any
import importlib

from docustruct.model.base import BaseOcrModel
from docustruct.model.registry import MODEL_REGISTRY, get_model_config, get_recommended_model


class ModelFactory:
    """Factory for creating OCR model instances"""
    
    _model_class_cache: Dict[str, type] = {}
    
    @classmethod
    def create_model(
        cls,
        model_id: str,
        device: str = "cuda",
        **kwargs
    ) -> BaseOcrModel:
        """
        Create an OCR model instance
        
        Args:
            model_id: Model identifier from the registry
            device: Device to run the model on ('cuda' or 'cpu')
            **kwargs: Additional model-specific parameters
        
        Returns:
            Instance of the OCR model
        
        Raises:
            ValueError: If model_id is not found in registry
            ImportError: If model class cannot be imported
        """
        config = get_model_config(model_id)
        
        if config is None:
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )
        
        # Get model class
        model_class = cls._get_model_class(config.model_class)
        
        # Prepare initialization parameters
        init_params = {
            "device": device,
            **kwargs
        }
        
        # Add model path if specified
        if config.model_path:
            init_params["model_path"] = config.model_path
        
        # Add API endpoint if specified
        if config.api_endpoint:
            init_params["api_endpoint"] = config.api_endpoint
        
        # Create and return model instance
        return model_class(**init_params)
    
    @classmethod
    def create_recommended_model(
        cls,
        language: Optional[str] = None,
        document_type: Optional[str] = None,
        quality: str = "high",
        device: str = "cuda",
        allow_commercial: bool = True,
        **kwargs
    ) -> BaseOcrModel:
        """
        Create a recommended OCR model based on requirements
        
        Args:
            language: Target language code (e.g., 'ar', 'en')
            document_type: Type of document ('handwritten', 'table', etc.)
            quality: Quality preference ('highest', 'high', 'balanced', 'fast')
            device: Device to run the model on
            allow_commercial: Whether to allow commercial API models
            **kwargs: Additional model-specific parameters
        
        Returns:
            Instance of the recommended OCR model
        """
        model_id = get_recommended_model(
            language=language,
            document_type=document_type,
            quality=quality,
            allow_commercial=allow_commercial
        )
        
        return cls.create_model(model_id, device=device, **kwargs)
    
    @classmethod
    def _get_model_class(cls, class_name: str) -> type:
        """
        Get model class by name, with caching
        
        Args:
            class_name: Name of the model class
        
        Returns:
            Model class
        
        Raises:
            ImportError: If class cannot be imported
        """
        if class_name in cls._model_class_cache:
            return cls._model_class_cache[class_name]
        
        # Map class names to module paths
        class_to_module = {
            "Qwen2VLModel": "docustruct.model.qwen2vl",
            "DotsOcrModel": "docustruct.model.dots_ocr",
            "QariOcrModel": "docustruct.model.qari_ocr",
            "DimiArabicOcrModel": "docustruct.model.dimi_arabic_ocr",
            "OcrRl2Model": "docustruct.model.ocr_rl2",
            "TrOcrArabicModel": "docustruct.model.trocr_arabic",
            "PaddleOcrArabicModel": "docustruct.model.paddle_ocr",
            "SuryaOcrModel": "docustruct.model.surya_ocr",
            "EasyOcrModel": "docustruct.model.easy_ocr",
            "TesseractModel": "docustruct.model.tesseract",
            "MistralOcrModel": "docustruct.model.mistral_ocr",
            "OpenAIGPT4OModel": "docustruct.model.openai_gpt4o",
            "HfOcrModel": "docustruct.model.hf",
            "VllmOcrModel": "docustruct.model.vllm",
        }
        
        module_path = class_to_module.get(class_name)
        
        if module_path is None:
            raise ImportError(f"Unknown model class: {class_name}")
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            cls._model_class_cache[class_name] = model_class
            return model_class
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_path}: {e}"
            )


# Convenience functions
def create_model(model_id: str, device: str = "cuda", **kwargs) -> BaseOcrModel:
    """Create an OCR model instance"""
    return ModelFactory.create_model(model_id, device, **kwargs)


def create_recommended_model(
    language: Optional[str] = None,
    document_type: Optional[str] = None,
    quality: str = "high",
    device: str = "cuda",
    **kwargs
) -> BaseOcrModel:
    """Create a recommended OCR model"""
    return ModelFactory.create_recommended_model(
        language, document_type, quality, device, **kwargs
    )
