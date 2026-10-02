#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

os.environ.setdefault("HF_HOME", "/models")
os.environ.setdefault("TRANSFORMERS_CACHE", "/models")
os.environ.setdefault("TORCH_HOME", "/models")

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime guard
    np = None

try:
    import cv2
except Exception:  # pragma: no cover - runtime guard
    cv2 = None

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception:  # pragma: no cover - runtime guard
    Image = None
    ImageFilter = None
    ImageOps = None

try:
    import torch
except Exception:  # pragma: no cover - runtime guard
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except Exception:  # pragma: no cover - runtime guard
    AutoModelForCausalLM = None
    AutoProcessor = None

try:
    import easyocr
except Exception:  # pragma: no cover - runtime guard
    easyocr = None

MODEL_CANDIDATES = [
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large",
]
CACHE_DIR = Path("/models")
DEFAULT_OUTPUT_DIR = Path("/app/output")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def resolve_output_dir() -> Path:
    candidates = [DEFAULT_OUTPUT_DIR, Path.cwd() / "output"]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path.cwd() / "output"


def normalize_text(raw_text: Optional[str]) -> str:
    if raw_text is None:
        return ""

    text = str(raw_text).replace("\x00", "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("<OCR>", " ").replace("</s>", " ")
    text = text.replace("[CLS]", " ").replace("[SEP]", " ")

    for marker in ("TEXT:", "OCR:", "OUTPUT:", "PREDICTED TEXT:"):
        upper_marker = marker.upper()
        if upper_marker in text.upper():
            text = text.upper().split(upper_marker, 1)[1]
            break

    text = text.replace("_", " ")
    text = re.sub(r"[\u2010-\u2015\u2212]+", "-", text)
    text = re.sub(r"[^A-Za-z0-9\s.,:/%&()#=+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.upper()
    text = re.sub(r"\s+", " ", text).strip()

    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def calculate_confidence(text: str, raw_score: Optional[float] = None) -> float:
    if not text or not text.strip():
        return 0.0

    cleaned = normalize_text(text)
    if not cleaned:
        return 0.0

    score = 0.0
    score += 0.20

    text_length = len(cleaned)
    alpha_numeric = sum(1 for ch in cleaned if ch.isalnum())
    alpha_ratio = alpha_numeric / max(1, text_length)

    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    word_count = len(words)
    if word_count:
        avg_word_length = sum(len(word) for word in words) / word_count
        score += min(0.22, (word_count / 12.0) * 0.12)
        score += 0.10 if 2 <= avg_word_length <= 8 else 0.0
    else:
        score -= 0.25

    score += min(0.28, (text_length / 80.0) * 0.20)
    score += min(0.18, max(0.0, alpha_ratio - 0.3) * 0.75)

    weird_char_count = sum(
        1 for ch in cleaned if not ch.isalnum() and not ch.isspace() and ch not in ".,:/%&()#=+-"
    )
    weird_ratio = weird_char_count / max(1, text_length)
    score -= min(0.30, weird_ratio * 1.3)

    if text_length < 4:
        score -= 0.25
    if word_count < 2 and text_length < 10:
        score -= 0.20
    if re.fullmatch(r"(?:[A-Z0-9]{1,3}\s*){3,}", cleaned) and len(cleaned) < 18:
        score -= 0.20

    if raw_score is not None:
        raw_value = max(0.0, min(1.0, float(raw_score)))
        score = (score * 0.75) + (raw_value * 0.25)

    score = max(0.0, min(1.0, score))
    return round(score, 4)


def detect_device() -> str:
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(image_path: Path) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_supported_image(image_path):
        raise ValueError(f"Unsupported format: {image_path.suffix}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=image.getchannel("A"))
        image = background
    elif image.mode == "L":
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image


def _model_fallback_text(raw_output: Any) -> str:
    if raw_output is None:
        return ""
    if isinstance(raw_output, (list, tuple)):
        raw_output = " ".join(str(part) for part in raw_output if part)
    text = str(raw_output)
    text = text.replace("<OCR>", " ").replace("</s>", " ")
    text = text.replace("<s>", " ").replace("</s>", " ")
    return normalize_text(text)


def run_florence_ocr(image: Image.Image, device: str) -> Tuple[str, float]:
    if torch is None or AutoProcessor is None or AutoModelForCausalLM is None:
        return "", 0.0

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        device_obj = torch.device(device)
        dtype = torch.float16 if device_obj.type == "cuda" else torch.float32

        last_error = None
        for model_id in MODEL_CANDIDATES:
            try:
                processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=str(CACHE_DIR),
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                model.to(device_obj)
                model.eval()

                inputs = processor(images=image, text="<OCR>", return_tensors="pt")
                inputs = {key: value.to(device_obj) if hasattr(value, "to") else value for key, value in inputs.items()}

                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=128,
                        do_sample=False,
                    )

                raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                extracted = _model_fallback_text(raw_output)
                if not extracted:
                    return "", 0.0
                confidence = calculate_confidence(extracted)
                return extracted, confidence
            except Exception as exc:  # pragma: no cover - fallback path
                last_error = exc
                continue

        if last_error is not None:
            return "", 0.0
        return "", 0.0
    except Exception:
        return "", 0.0


def get_easyocr_reader() -> Any:
    if easyocr is None:
        raise RuntimeError("EasyOCR is not available")

    if not hasattr(get_easyocr_reader, "reader"):
        gpu_enabled = torch is not None and torch.cuda.is_available()
        get_easyocr_reader.reader = easyocr.Reader(["en"], gpu=gpu_enabled, model_storage_directory=str(CACHE_DIR / "easyocr"))
    return get_easyocr_reader.reader


def run_easyocr(image: Image.Image) -> Tuple[str, float]:
    if easyocr is None:
        return "", 0.0

    try:
        reader = get_easyocr_reader()
        image_array = np.asarray(image) if np is not None else None
        if image_array is None:
            return "", 0.0
        results = reader.readtext(image_array, detail=0, paragraph=True)
        if not results:
            return "", 0.0

        blocks = [str(item).strip() for item in results if str(item).strip()]
        text = " ".join(blocks)
        cleaned = normalize_text(text)
        if not cleaned:
            return "", 0.0
        return cleaned, calculate_confidence(cleaned)
    except Exception:
        return "", 0.0


def detect_text(image_path: Path) -> Tuple[str, float]:
    try:
        image = load_image(image_path)
    except Exception:
        return "", 0.0

    try:
        device = detect_device()
        primary_text, primary_confidence = run_florence_ocr(image, device)
        if primary_text:
            return primary_text, primary_confidence
    except Exception:
        pass

    fallback_text, fallback_confidence = run_easyocr(image)
    if fallback_text:
        return fallback_text, fallback_confidence

    return "", 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR pipeline with VLM-first fallback")
    parser.add_argument("--input-image", required=True, help="Path to the input image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_image).expanduser().resolve()

    output_dir = resolve_output_dir()
    output_path = output_dir / f"{input_path.stem}_output.json"

    result = {"text": "", "confidence": 0.0}
    try:
        if input_path.exists() and is_supported_image(input_path):
            detected_text, confidence = detect_text(input_path)
            if detected_text:
                result = {"text": detected_text, "confidence": confidence}
        else:
            result = {"text": "", "confidence": 0.0}
    except Exception:
        result = {"text": "", "confidence": 0.0}

    output_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"Saved OCR result to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
