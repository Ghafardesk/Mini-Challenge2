#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
os.environ.setdefault("TORCH_HOME", "/models")

try:
    from PIL import Image
except Exception:  # pragma: no cover - runtime guard
    Image = None

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - runtime guard
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - runtime guard
    AutoTokenizer = None

MODEL_PATH_OR_NAME = os.getenv(
    "HF_MODEL_ID",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
)
CACHE_DIR = Path(os.getenv("HF_HOME", "/models"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_image(image_path: Path) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    image = Image.open(image_path)
    image = image.convert("RGB")
    return image


class ImageInference:
    def __init__(self, model_name: str = MODEL_PATH_OR_NAME):
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM is required for OCR inference.")
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for OCR inference.")

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_image_output(self, image: Image.Image) -> str:
        messages = [{
            "role": "user",
            "content": (
                "Act as an OCR assistant. Analyze the provided <|image|> image and:\n"
                "1. Identify and transcribe all visible text in the image exactly as it appears.\n"
                "2. Preserve the original line breaks, spacing, and formatting from the image.\n"
                "3. Output only the transcribed text, line by line, without adding any commentary or explanations or special characters.\n"
            ),
        }]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(max_tokens=512, temperature=0.7)
        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
        )

        generated_text = outputs[0].outputs[0].text if outputs else "No output generated."
        return generated_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR pipeline using AMD ROCm vLLM + Llama 3.2 Vision")
    parser.add_argument("--input-image", required=True, help="Path to the input image")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for OCR JSON output")
    parser.add_argument("--model", default=MODEL_PATH_OR_NAME, help="Hugging Face model ID or local path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input image does not exist: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SystemExit(f"Unsupported image format: {input_path.suffix}")

    output_path = output_dir / f"{input_path.stem}_output.json"

    try:
        image = load_image(input_path)
        inference = ImageInference(model_name=args.model)
        detected_text = inference.generate_image_output(image)
        result = {
            "text": detected_text,
            "confidence": 1.0 if detected_text.strip() else 0.0,
            "model": args.model,
        }
    except Exception as exc:
        print(f"OCR failed: {exc}", file=sys.stderr)
        result = {"text": "", "confidence": 0.0, "model": args.model}

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved OCR result to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
