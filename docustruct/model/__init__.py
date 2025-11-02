"""
DocuStruct Model Module

This module provides OCR model implementations and management.
Includes both the original InferenceManager and the new multi-model architecture.
"""

from typing import List

# Original imports for backward compatibility
from docustruct.model.hf import load_model, generate_hf
from docustruct.model.schema import BatchInputItem, BatchOutputItem, OcrResult
from docustruct.model.vllm import generate_vllm
from docustruct.output import parse_markdown, parse_html, parse_chunks, extract_images

# New multi-model architecture imports
from docustruct.model.base import BaseOcrModel
from docustruct.model.registry import (
    MODEL_REGISTRY,
    ModelConfig,
    ModelTier,
    get_model_config,
    list_models,
    get_recommended_model,
)
from docustruct.model.factory import ModelFactory, create_model, create_recommended_model

# Import all model implementations (with graceful fallbacks)
try:
    from docustruct.model.qwen2vl import Qwen2VLModel
except ImportError:
    Qwen2VLModel = None

try:
    from docustruct.model.dots_ocr import DotsOcrModel
except ImportError:
    DotsOcrModel = None

try:
    from docustruct.model.qari_ocr import QariOcrModel
except ImportError:
    QariOcrModel = None

try:
    from docustruct.model.dimi_arabic_ocr import DimiArabicOcrModel
except ImportError:
    DimiArabicOcrModel = None

try:
    from docustruct.model.ocr_rl2 import OcrRl2Model
except ImportError:
    OcrRl2Model = None

try:
    from docustruct.model.easy_ocr import EasyOcrModel
except ImportError:
    EasyOcrModel = None

try:
    from docustruct.model.tesseract import TesseractModel
except ImportError:
    TesseractModel = None

try:
    from docustruct.model.paddle_ocr import PaddleOcrArabicModel
except ImportError:
    PaddleOcrArabicModel = None

try:
    from docustruct.model.trocr_arabic import TrOcrArabicModel
except ImportError:
    TrOcrArabicModel = None

try:
    from docustruct.model.surya_ocr import SuryaOcrModel
except ImportError:
    SuryaOcrModel = None

try:
    from docustruct.model.openai_gpt4o import OpenAIGPT4OModel
except ImportError:
    OpenAIGPT4OModel = None

try:
    from docustruct.model.mistral_ocr import MistralOcrModel
except ImportError:
    MistralOcrModel = None


class InferenceManager:
    """
    Original InferenceManager for backward compatibility.
    Supports 'vllm' and 'hf' methods.
    """
    
    def __init__(self, method: str = "vllm"):
        assert method in ("vllm", "hf"), "method must be 'vllm' or 'hf'"
        self.method = method

        if method == "hf":
            self.model = load_model()
        else:
            self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        output_kwargs = {}
        if "include_images" in kwargs:
            output_kwargs["include_images"] = kwargs.pop("include_images")
        if "include_headers_footers" in kwargs:
            output_kwargs["include_headers_footers"] = kwargs.pop(
                "include_headers_footers"
            )

        if self.method == "vllm":
            results = generate_vllm(
                batch, max_output_tokens=max_output_tokens, **kwargs
            )
        else:
            results = generate_hf(
                batch, self.model, max_output_tokens=max_output_tokens, **kwargs
            )

        output = []
        for result, input_item in zip(results, batch):
            chunks = parse_chunks(result.raw, input_item.image)
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw, **output_kwargs),
                    html=parse_html(result.raw, **output_kwargs),
                    chunks=chunks,
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count,
                    images=extract_images(result.raw, chunks, input_item.image),
                    error=result.error,
                )
            )
        return output


__all__ = [
    # Original exports
    "InferenceManager",
    "load_model",
    "generate_hf",
    "generate_vllm",
    "BatchInputItem",
    "BatchOutputItem",
    
    # Base classes
    "BaseOcrModel",
    "OcrResult",
    
    # Registry
    "MODEL_REGISTRY",
    "ModelConfig",
    "ModelTier",
    "get_model_config",
    "list_models",
    "get_recommended_model",
    
    # Factory
    "ModelFactory",
    "create_model",
    "create_recommended_model",
    
    # Model implementations
    "Qwen2VLModel",
    "DotsOcrModel",
    "QariOcrModel",
    "DimiArabicOcrModel",
    "OcrRl2Model",
    "EasyOcrModel",
    "TesseractModel",
    "PaddleOcrArabicModel",
    "TrOcrArabicModel",
    "SuryaOcrModel",
    "OpenAIGPT4OModel",
    "MistralOcrModel",
]
