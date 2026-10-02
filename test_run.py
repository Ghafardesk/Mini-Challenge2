#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_image(path: Path) -> None:
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()

    # Simulated road sign
    sign_color = (222, 48, 48)
    draw.rounded_rectangle((100, 80, 980, 620), radius=40, fill=sign_color)
    draw.rounded_rectangle((150, 130, 930, 570), radius=30, fill=(255, 255, 255))
    draw.text((220, 210), "STOP", fill=(0, 0, 0), font=font_large)
    draw.text((220, 415), "CARS ONLY", fill=(0, 0, 0), font=font_medium)

    # Simulated license plate
    plate_color = (220, 220, 220)
    draw.rounded_rectangle((560, 70, 1100, 220), radius=18, fill=plate_color)
    draw.text((600, 110), "ABC1234", fill=(0, 0, 0), font=font_medium)

    image.save(path)


def main() -> int:
    sample_path = INPUT_DIR / "sample_road_sign.jpg"
    create_sample_image(sample_path)

    cmd = [sys.executable, str(ROOT / "app.py"), "--input-image", str(sample_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    output_candidates = [
        Path("/app/output/sample_road_sign_output.json"),
        OUTPUT_DIR / "sample_road_sign_output.json",
    ]
    output_path = next((candidate for candidate in output_candidates if candidate.exists()), output_candidates[0])
    if not output_path.exists():
        raise FileNotFoundError(f"Expected output JSON at {output_path}")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if set(payload.keys()) != {"text", "confidence"}:
        raise ValueError(f"Unexpected JSON schema: {payload}")

    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
