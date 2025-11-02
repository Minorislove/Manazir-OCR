"""
Model Registry for DocuStruct OCR Framework

This module defines all available OCR models and their configurations.
It provides a centralized registry for model selection and management.
"""

from typing import Dict, List, Optional, Any
from enum import Enum


class ModelTier(str, Enum):
    """Model tiers based on priority and use cases"""
    TIER_1_PRIMARY = "tier_1_primary"
    TIER_2_SPECIALIZED = "tier_2_specialized"
    TIER_3_LIGHTWEIGHT = "tier_3_lightweight"
    TIER_4_BASELINE = "tier_4_baseline"
    COMMERCIAL = "commercial"


class ModelConfig:
    """Configuration for an OCR model"""
    
    def __init__(
        self,
        model_id: str,
        model_class: str,
        display_name: str,
        languages: List[str],
        strengths: List[str],
        license: str,
        tier: ModelTier,
        commercial_use: bool,
        model_path: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        requires_gpu: bool = True,
        model_size: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_class = model_class
        self.display_name = display_name
        self.languages = languages
        self.strengths = strengths
        self.license = license
        self.tier = tier
        self.commercial_use = commercial_use
        self.model_path = model_path
        self.api_endpoint = api_endpoint
        self.requires_gpu = requires_gpu
        self.model_size = model_size
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_id": self.model_id,
            "model_class": self.model_class,
            "display_name": self.display_name,
            "languages": self.languages,
            "strengths": self.strengths,
            "license": self.license,
            "tier": self.tier.value,
            "commercial_use": self.commercial_use,
            "model_path": self.model_path,
            "api_endpoint": self.api_endpoint,
            "requires_gpu": self.requires_gpu,
            "model_size": self.model_size,
            "description": self.description,
        }


# Model Registry
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # ========== TIER 1: PRIMARY MULTILINGUAL MODELS ==========
    
    "qwen2_vl_2b": ModelConfig(
        model_id="qwen2_vl_2b",
        model_class="Qwen2VLModel",
        display_name="Qwen2-VL 2B",
        languages=["ar", "en", "zh", "ja", "ko", "vi", "es", "fr", "de", "it", "pt", "ru"],
        strengths=["general_vlm", "multilingual", "arabic", "document_understanding"],
        license="apache-2.0",
        tier=ModelTier.TIER_1_PRIMARY,
        commercial_use=True,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        requires_gpu=True,
        model_size="2B",
        description="Versatile vision-language model with explicit Arabic support. Excellent for general OCR and document understanding.",
    ),
    
    "dots_ocr": ModelConfig(
        model_id="dots_ocr",
        model_class="DotsOcrModel",
        display_name="dots.ocr",
        languages=["*"],  # 100+ languages
        strengths=["multilingual", "tables", "formulas", "layout", "reading_order"],
        license="mit",
        tier=ModelTier.TIER_1_PRIMARY,
        commercial_use=True,
        model_path="rednote-hilab/dots.ocr",
        requires_gpu=True,
        model_size="1.7B",
        description="SOTA multilingual OCR supporting 100+ languages. Best for complex documents with tables and formulas.",
    ),
    
    # ========== TIER 2: LANGUAGE-SPECIFIC MODELS ==========
    
    "qari_ocr": ModelConfig(
        model_id="qari_ocr",
        model_class="QariOcrModel",
        display_name="Qari-OCR",
        languages=["ar"],
        strengths=["arabic_printed", "high_accuracy", "full_page"],
        license="apache-2.0",
        tier=ModelTier.TIER_2_SPECIALIZED,
        commercial_use=True,
        model_path="NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct",
        requires_gpu=True,
        model_size="2B",
        description="Specialized Arabic OCR with 98.1% character accuracy. Best for printed Arabic text.",
    ),
    
    "dimi_arabic_ocr": ModelConfig(
        model_id="dimi_arabic_ocr",
        model_class="DimiArabicOcrModel",
        display_name="DIMI-Arabic-OCR",
        languages=["ar"],
        strengths=["arabic_documents", "markdown_output", "structured_extraction"],
        license="apache-2.0",
        tier=ModelTier.TIER_2_SPECIALIZED,
        commercial_use=True,
        model_path="Ahmed-Zaky/DIMI-Arabic-OCR-markdown",
        requires_gpu=True,
        model_size="N/A",
        description="Arabic OCR model specialized for converting documents to markdown format. By Ahmed Zaky.",
    ),
    
    "ocr_rl2": ModelConfig(
        model_id="ocr_rl2",
        model_class="OcrRl2Model",
        display_name="OCR-RL2",
        languages=["ar"],
        strengths=["arabic_ocr", "reinforcement_learning"],
        license="apache-2.0",
        tier=ModelTier.TIER_2_SPECIALIZED,
        commercial_use=True,
        model_path="Mohamed-Zayton/ocr_rl2",
        requires_gpu=True,
        model_size="N/A",
        description="Arabic OCR model trained with reinforcement learning. By Mohamed Zayton.",
    ),
    
    "trocr_arabic": ModelConfig(
        model_id="trocr_arabic",
        model_class="TrOcrArabicModel",
        display_name="TrOCR Arabic",
        languages=["ar", "en"],
        strengths=["arabic_handwriting", "bilingual"],
        license="mit",
        tier=ModelTier.TIER_2_SPECIALIZED,
        commercial_use=True,
        model_path="David-Magdy/TR_OCR_LARGE",
        requires_gpu=True,
        model_size="Large",
        description="Specialized for handwritten Arabic text recognition.",
    ),

    # Additional Arabic models discovered
    "qwen2_5_vl_7b_arabic": ModelConfig(
        model_id="qwen2_5_vl_7b_arabic",
        model_class="Qwen2VLModel",
        display_name="Qwen2.5-VL 7B Arabic OCR",
        languages=["ar"],
        strengths=["arabic", "document_understanding", "image_to_text"],
        license="apache-2.0",
        tier=ModelTier.TIER_1_PRIMARY,
        commercial_use=True,
        model_path="loay/Arabic-OCR-Qwen2.5-VL-7B-Vision",
        requires_gpu=True,
        model_size="7B",
        description="Arabic OCR using Qwen2.5-VL 7B Vision fine-tune.",
    ),

    "paddle_ocr_arabic_v4": ModelConfig(
        model_id="paddle_ocr_arabic_v4",
        model_class="PaddleOcrArabicModel",
        display_name="PaddleOCR Arabic (PP-OCRv4)",
        languages=["ar"],
        strengths=["lightweight", "fast", "mobile"],
        license="apache-2.0",
        tier=ModelTier.TIER_3_LIGHTWEIGHT,
        commercial_use=True,
        model_path="cycloneboy/arabic_PP-OCRv4_rec_infer",
        requires_gpu=False,
        model_size="Small",
        description="PaddleOCR Arabic recognition with PP-OCRv4 weights (via PaddleOCR backend).",
    ),

    "surya_ocr_arabic": ModelConfig(
        model_id="surya_ocr_arabic",
        model_class="SuryaOcrModel",
        display_name="Surya OCR Arabic",
        languages=["ar"],
        strengths=["layout_analysis", "reading_order", "tables", "arabic"],
        license="gpl-3.0",
        tier=ModelTier.TIER_3_LIGHTWEIGHT,
        commercial_use=False,
        model_path="ketanmore/surya-ocr-arabic",
        requires_gpu=True,
        model_size="N/A",
        description="Surya OCR tuned for Arabic documents (GPL-3.0).",
    ),

    "qari_ocr_waraqon": ModelConfig(
        model_id="qari_ocr_waraqon",
        model_class="QariOcrModel",
        display_name="Qari-OCR (Waraqon Arabic HTML FT)",
        languages=["ar"],
        strengths=["arabic_printed", "high_accuracy", "html_output"],
        license="apache-2.0",
        tier=ModelTier.TIER_2_SPECIALIZED,
        commercial_use=True,
        model_path="FatimahEmadEldin/Waraqon-Arabic-OCR-HTML-Qari-Fine-Tuned",
        requires_gpu=True,
        model_size="2B",
        description="Fine-tuned Qari model for Arabic OCR to HTML.",
    ),
    
    # ========== TIER 3: LIGHTWEIGHT MODELS ==========
    
    "paddle_ocr_arabic": ModelConfig(
        model_id="paddle_ocr_arabic",
        model_class="PaddleOcrArabicModel",
        display_name="PaddleOCR Arabic",
        languages=["ar"],
        strengths=["lightweight", "fast", "mobile"],
        license="apache-2.0",
        tier=ModelTier.TIER_3_LIGHTWEIGHT,
        commercial_use=True,
        model_path="PaddlePaddle/arabic_PP-OCRv3_mobile_rec",
        requires_gpu=False,
        model_size="7.8MB",
        description="Ultra-lightweight Arabic OCR. Perfect for mobile and edge devices.",
    ),
    
    "surya_ocr": ModelConfig(
        model_id="surya_ocr",
        model_class="SuryaOcrModel",
        display_name="Surya OCR",
        languages=["*"],  # 90+ languages
        strengths=["layout_analysis", "reading_order", "tables", "multilingual"],
        license="gpl-3.0",
        tier=ModelTier.TIER_3_LIGHTWEIGHT,
        commercial_use=False,  # GPL requires careful consideration
        model_path="hesham-haroon/surya",
        requires_gpu=True,
        model_size="N/A",
        description="Comprehensive document OCR toolkit with layout analysis. Note: GPL-3.0 license.",
    ),
    
    # ========== TIER 4: BASELINE MODELS ==========
    
    "easy_ocr": ModelConfig(
        model_id="easy_ocr",
        model_class="EasyOcrModel",
        display_name="EasyOCR",
        languages=["*"],  # 80+ languages
        strengths=["easy_integration", "baseline", "prototyping"],
        license="apache-2.0",
        tier=ModelTier.TIER_4_BASELINE,
        commercial_use=True,
        model_path="easyocr",
        requires_gpu=False,
        model_size="N/A",
        description="Simple and easy-to-use OCR for quick prototyping.",
    ),
    
    "tesseract": ModelConfig(
        model_id="tesseract",
        model_class="TesseractModel",
        display_name="Tesseract OCR",
        languages=["*"],  # 100+ languages
        strengths=["lightweight", "mature", "baseline", "cpu_only"],
        license="apache-2.0",
        tier=ModelTier.TIER_4_BASELINE,
        commercial_use=True,
        model_path="tesseract",
        requires_gpu=False,
        model_size="Small",
        description="Traditional OCR engine. Lightweight and mature baseline option.",
    ),
    
    # ========== COMMERCIAL API MODELS ==========
    
    "mistral_ocr": ModelConfig(
        model_id="mistral_ocr",
        model_class="MistralOcrModel",
        display_name="Mistral OCR",
        languages=["*"],
        strengths=["enterprise", "complex_documents", "high_accuracy"],
        license="commercial",
        tier=ModelTier.COMMERCIAL,
        commercial_use=True,
        api_endpoint="https://api.mistral.ai/v1/ocr",
        requires_gpu=False,
        model_size="API",
        description="Enterprise-grade OCR API. Excellent for complex documents.",
    ),
    
    "openai_gpt4o": ModelConfig(
        model_id="openai_gpt4o",
        model_class="OpenAIGPT4OModel",
        display_name="OpenAI GPT-4o",
        languages=["*"],
        strengths=["enterprise", "multimodal", "high_accuracy"],
        license="commercial",
        tier=ModelTier.COMMERCIAL,
        commercial_use=True,
        api_endpoint="https://api.openai.com/v1/chat/completions",
        requires_gpu=False,
        model_size="API",
        description="OpenAI's multimodal model with OCR capabilities.",
    ),
    
    # ========== ORIGINAL DOCUSTRUCT MODEL ==========
    
    "docustruct_hf": ModelConfig(
        model_id="docustruct_hf",
        model_class="HfOcrModel",
        display_name="DocuStruct HF (Original)",
        languages=["en", "zh"],
        strengths=["original_model", "document_structure"],
        license="apache-2.0",
        tier=ModelTier.TIER_1_PRIMARY,
        commercial_use=True,
        model_path="original",
        requires_gpu=True,
        model_size="N/A",
        description="Original DocuStruct model based on Qwen3VL.",
    ),
    
    "docustruct_vllm": ModelConfig(
        model_id="docustruct_vllm",
        model_class="VllmOcrModel",
        display_name="DocuStruct vLLM (Original)",
        languages=["en", "zh"],
        strengths=["original_model", "vllm_server", "fast_inference"],
        license="apache-2.0",
        tier=ModelTier.TIER_1_PRIMARY,
        commercial_use=True,
        api_endpoint="vllm_server",
        requires_gpu=True,
        model_size="N/A",
        description="Original DocuStruct model with vLLM server for fast inference.",
    ),
}


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get model configuration by ID"""
    return MODEL_REGISTRY.get(model_id)


def list_models(
    tier: Optional[ModelTier] = None,
    language: Optional[str] = None,
    commercial_only: bool = False,
) -> List[ModelConfig]:
    """List available models with optional filtering"""
    models = list(MODEL_REGISTRY.values())
    
    if tier:
        models = [m for m in models if m.tier == tier]
    
    if language:
        models = [m for m in models if "*" in m.languages or language in m.languages]
    
    if commercial_only:
        models = [m for m in models if m.commercial_use]
    
    return models


def get_recommended_model(
    language: Optional[str] = None,
    document_type: Optional[str] = None,
    quality: str = "high",
    allow_commercial: bool = True,
) -> str:
    """
    Get recommended model ID based on requirements
    
    Args:
        language: Target language code (e.g., 'ar', 'en')
        document_type: Type of document ('handwritten', 'table', 'formula', etc.)
        quality: Quality preference ('highest', 'high', 'balanced', 'fast')
        allow_commercial: Whether to allow commercial API models
    
    Returns:
        Recommended model ID
    """
    def _allowed(mid: str) -> bool:
        cfg = MODEL_REGISTRY.get(mid)
        if cfg is None:
            return False
        if allow_commercial:
            return True
        return cfg.tier != ModelTier.COMMERCIAL

    # Arabic-specific routing
    if language == "ar":
        if quality == "highest":
            return "qari_ocr" if _allowed("qari_ocr") else "dots_ocr"
        elif document_type == "handwritten":
            return "trocr_arabic" if _allowed("trocr_arabic") else "qari_ocr"
        elif quality == "fast":
            return "paddle_ocr_arabic" if _allowed("paddle_ocr_arabic") else "easy_ocr"
        else:
            return "qari_ocr" if _allowed("qari_ocr") else "dots_ocr"
    
    # Document type routing
    if document_type in ["table", "formula", "layout", "complex"]:
        return "dots_ocr" if _allowed("dots_ocr") else "easy_ocr"
    
    # Multilingual routing
    if language in ["en", "zh", "ja", "ko", "vi", "es", "fr", "de"]:
        if quality in ["highest", "high"]:
            return "qwen2_vl_2b" if _allowed("qwen2_vl_2b") else "dots_ocr"
        else:
            return "dots_ocr" if _allowed("dots_ocr") else "easy_ocr"
    
    # Default multilingual fallback
    return "dots_ocr" if _allowed("dots_ocr") else "easy_ocr"
