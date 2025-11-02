from typing import List
import logging

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from docustruct.model.schema import BatchInputItem, GenerationResult
from docustruct.model.util import scale_to_fit
from docustruct.prompts import PROMPT_MAPPING
from docustruct.settings import settings

logger = logging.getLogger(__name__)


def generate_hf(
    batch: List[BatchInputItem], model, max_output_tokens=None, **kwargs
) -> List[GenerationResult]:
    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    messages = [process_batch_element(item, model.processor) for item in batch]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = model.processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_output_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]
    return results


def process_batch_element(item: BatchInputItem, processor):
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        prompt = PROMPT_MAPPING[prompt_type]

    content = []
    image = scale_to_fit(item.image)  # Guarantee max size
    content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})
    message = {"role": "user", "content": content}
    return message


def load_model():
    device_map = "auto"
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}

    dtype = settings.TORCH_DTYPE
    if not torch.cuda.is_available():
        # Fallback to fp32 on CPU to maximize compatibility
        dtype = torch.float32

    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    # Try Qwen3VL first (for legacy/private models)
    try:
        logger.info(f"Attempting to load Qwen3VL model from {settings.MODEL_CHECKPOINT}...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.MODEL_CHECKPOINT, **kwargs
        )
        model = model.eval()
        processor = Qwen3VLProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
        model.processor = processor
        logger.info("Successfully loaded Qwen3VL model")
        return model
    except (OSError, ValueError, Exception) as e:
        # Fallback to Qwen2VL if Qwen3VL fails
        logger.warning(
            f"Failed to load Qwen3VL model: {e}. "
            f"Falling back to Qwen2VL architecture..."
        )
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                settings.MODEL_CHECKPOINT, **kwargs
            )
            model = model.eval()
            processor = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
            model.processor = processor
            logger.info("Successfully loaded Qwen2VL model")
            return model
        except Exception as e2:
            raise OSError(
                f"Failed to load both Qwen3VL and Qwen2VL models from {settings.MODEL_CHECKPOINT}. "
                f"Qwen3VL error: {e}. Qwen2VL error: {e2}"
            ) from e2
