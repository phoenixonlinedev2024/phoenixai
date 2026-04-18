"""Image generation tools — HuggingFace Inference API (free tier) + local Stable Diffusion."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry

from jarvis.config import cfg


def _generate_image_hf(prompt: str, model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                        output_path: str = "/tmp/jarvis_image.png") -> str:
    """Generate image via HuggingFace Inference API (free tier, no key needed for some models)."""
    try:
        import requests
        headers = {}
        if cfg.HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {cfg.HF_API_TOKEN}"
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        resp = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=60)
        if resp.status_code == 200:
            Path(output_path).write_bytes(resp.content)
            return f"Image generated and saved to {output_path} ({len(resp.content)} bytes)"
        return f"HuggingFace API error {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return f"Image generation error: {exc}"


def _generate_image_local(prompt: str, output_path: str = "/tmp/jarvis_image.png",
                           steps: int = 20, width: int = 512, height: int = 512) -> str:
    """Generate image locally using diffusers (requires GPU or slow CPU)."""
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        image = pipe(prompt, num_inference_steps=steps, width=width, height=height).images[0]
        image.save(output_path)
        return f"Image generated locally and saved to {output_path}"
    except ImportError:
        return "diffusers not installed. Run: pip install diffusers transformers accelerate torch"
    except Exception as exc:
        return f"Local image generation error: {exc}"


def _describe_image(image_path: str) -> str:
    """Describe or analyse an image using Claude vision."""
    try:
        import anthropic
        import base64
        data = Path(image_path).read_bytes()
        b64 = base64.standard_b64encode(data).decode()
        ext = Path(image_path).suffix.lstrip(".").lower()
        media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                     "gif": "image/gif", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/png")
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }],
        )
        return resp.content[0].text
    except Exception as exc:
        return f"Vision error: {exc}"


def _analyze_image(image_path: str, question: str) -> str:
    """Answer a specific question about an image using Claude vision."""
    try:
        import anthropic, base64
        data = Path(image_path).read_bytes()
        b64 = base64.standard_b64encode(data).decode()
        ext = Path(image_path).suffix.lstrip(".").lower()
        media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                     "gif": "image/gif", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/png")
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": question},
                ],
            }],
        )
        return resp.content[0].text
    except Exception as exc:
        return f"Vision analysis error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="generate_image",
        description="Generate an image from a text prompt using HuggingFace Inference API (free). Saves as PNG.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image generation prompt"},
                "model": {"type": "string", "default": "stabilityai/stable-diffusion-xl-base-1.0"},
                "output_path": {"type": "string", "default": "/tmp/jarvis_image.png"},
            },
            "required": ["prompt"],
        },
        fn=_generate_image_hf,
        category="media",
    ))

    registry.register(Tool(
        name="generate_image_local",
        description="Generate an image locally using Stable Diffusion (requires pip install diffusers torch).",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "output_path": {"type": "string", "default": "/tmp/jarvis_image.png"},
                "steps": {"type": "integer", "default": 20},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
            },
            "required": ["prompt"],
        },
        fn=_generate_image_local,
        category="media",
    ))

    registry.register(Tool(
        name="describe_image",
        description="Describe and analyse the contents of an image file using Claude vision.",
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to image file"},
            },
            "required": ["image_path"],
        },
        fn=_describe_image,
        category="media",
    ))

    registry.register(Tool(
        name="analyze_image",
        description="Answer a specific question about an image file using Claude vision.",
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "question": {"type": "string", "description": "Question to answer about the image"},
            },
            "required": ["image_path", "question"],
        },
        fn=_analyze_image,
        category="media",
    ))
