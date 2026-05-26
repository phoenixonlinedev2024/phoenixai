"""Tests for image_gen and creator (synthesise_tool)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()


# ══════════════════════════════════════════════════════════════════════════════
# image_gen
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.image_gen import (  # noqa: E402
    _generate_image_hf,
    _generate_image_local,
    _describe_image,
    _analyze_image,
)


def test_generate_image_hf_success(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.content = b"\x89PNG\r\n"  # 6 bytes of fake PNG
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(return_value=fake_resp)

    out_path = str(tmp_path / "img.png")
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("a cat", output_path=out_path)
    assert "saved to" in out
    assert Path(out_path).exists()


def test_generate_image_hf_includes_auth_header(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "my-hf-token")
    call_kwargs = {}
    fake_resp = MagicMock(status_code=200, content=b"bytes")
    fake_requests = MagicMock()

    def fake_post(url, headers=None, json=None, timeout=None):
        call_kwargs["headers"] = headers
        return fake_resp

    fake_requests.post = fake_post
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _generate_image_hf("robot", output_path="/tmp/x.png")
    assert "Authorization" in call_kwargs["headers"]
    assert "my-hf-token" in call_kwargs["headers"]["Authorization"]


def test_generate_image_hf_api_error(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    fake_resp = MagicMock(status_code=503, text="Service Unavailable")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("test")
    assert "503" in out


def test_generate_image_hf_exception(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(side_effect=RuntimeError("network dead"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("test")
    assert "Image generation error" in out


def test_generate_image_local_missing_diffusers():
    with patch.dict(sys.modules, {"diffusers": None, "torch": None}):
        out = _generate_image_local("a robot")
    assert "diffusers not installed" in out


def test_describe_image_calls_claude(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "test-key")

    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal valid header bytes

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="A grey robot standing in a field.")]
    fake_client.messages.create = MagicMock(return_value=fake_resp)

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        out = _describe_image(str(img_path))
    assert "robot" in out


def test_describe_image_exception():
    out = _describe_image("/tmp/definitely_missing_file_xyz.png")
    assert "Vision error" in out


def test_analyze_image_calls_claude_with_question(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "test-key")

    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff")  # JPEG header

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="Yes, there is a cat.")]
    fake_client.messages.create = MagicMock(return_value=fake_resp)

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        out = _analyze_image(str(img_path), "Is there a cat?")
    # Verify the question was passed to the API
    args = fake_client.messages.create.call_args
    user_content = args.kwargs["messages"][0]["content"]
    questions = [b["text"] for b in user_content if b["type"] == "text"]
    assert "Is there a cat?" in questions
    assert "cat" in out


def test_image_gen_register_tools():
    from jarvis.tools.image_gen import register_tools
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    register_tools(reg)
    names = {t.name for t in reg.all()}
    assert {"generate_image", "generate_image_local", "describe_image", "analyze_image"} <= names


# ══════════════════════════════════════════════════════════════════════════════
# creator — synthesise_tool
# ══════════════════════════════════════════════════════════════════════════════

from jarvis.tools.creator import synthesise_tool, _make_fn  # noqa: E402


def _valid_payload(name="test_tool"):
    return json.dumps({
        "name": name,
        "description": "A test tool.",
        "category": "general",
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        "python_code": "def run(**kwargs):\n    return 'ok'",
    })


@pytest.mark.asyncio
async def test_synthesise_tool_registers_and_returns(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=_valid_payload())]
    ))

    tool = await synthesise_tool("test capability", client, reg)
    assert tool is not None
    assert tool.name == "test_tool"
    assert reg.get("test_tool") is tool
    # File was persisted
    assert (tmp_path / "test_tool.py").exists()


@pytest.mark.asyncio
async def test_synthesise_tool_strips_code_fences(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    fenced = f"```json\n{_valid_payload('fenced_tool')}\n```"
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=fenced)]
    ))

    tool = await synthesise_tool("fenced capability", client, reg)
    assert tool is not None
    assert tool.name == "fenced_tool"


@pytest.mark.asyncio
async def test_synthesise_tool_api_error_returns_none(tmp_path, monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))

    tool = await synthesise_tool("anything", client, reg)
    assert tool is None
    assert "Tool synthesis error" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_synthesise_tool_malformed_json_returns_none(tmp_path, monkeypatch, capsys):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text="not json at all")]
    ))

    tool = await synthesise_tool("whatever", client, reg)
    assert tool is None


@pytest.mark.asyncio
async def test_synthesise_tool_fn_executes(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    payload = json.dumps({
        "name": "adder_tool",
        "description": "Adds two numbers.",
        "category": "general",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        "python_code": "def run(**kwargs):\n    return str(kwargs['a'] + kwargs['b'])",
    })
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=payload)]
    ))

    tool = await synthesise_tool("add numbers", client, reg)
    assert tool is not None
    assert tool.fn(a=3, b=4) == "7"


# ── _make_fn ─────────────────────────────────────────────────────────────────

def test_make_fn_compiles_and_runs():
    code = "def run(**kwargs):\n    return 'hello ' + kwargs['name']"
    fn = _make_fn(code)
    assert fn(name="World") == "hello World"


def test_make_fn_import_inside_body():
    code = "def run(**kwargs):\n    import math\n    return str(math.pi)"
    fn = _make_fn(code)
    assert "3.14" in fn()


def test_make_fn_syntax_error_raises():
    with pytest.raises(SyntaxError):
        _make_fn("def run(**kwargs):\n    return ??? bad syntax")


@pytest.mark.asyncio
async def test_synthesise_tool_code_fence_without_json_prefix(tmp_path, monkeypatch):
    """Branch 65->68: code fence content not starting with 'json' skips raw[4:]."""
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")

    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    # Fence is "```\n{...}" — split gives "\n{...}", doesn't start with "json"
    fenced = f"```\n{_valid_payload('no_json_prefix_tool')}\n```"
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=fenced)]
    ))

    tool = await synthesise_tool("no-json-prefix capability", client, reg)
    assert tool is not None
    assert tool.name == "no_json_prefix_tool"


# ── Additional image_gen tests ────────────────────────────────────────────────

def test_generate_image_local_success(tmp_path):
    fake_pipe = MagicMock()
    fake_image = MagicMock()
    fake_pipe.return_value.images = [fake_image]

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.float32 = "float32"

    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionPipeline.from_pretrained.return_value = MagicMock()
    fake_diffusers.StableDiffusionPipeline.from_pretrained.return_value.to.return_value = fake_pipe

    out_path = str(tmp_path / "local.png")
    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        out = _generate_image_local("a sunset", output_path=out_path)
    assert "saved to" in out or "generated" in out.lower()


def test_generate_image_local_exception_returns_error():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionPipeline.from_pretrained.side_effect = RuntimeError("GPU OOM")
    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        out = _generate_image_local("test prompt")
    assert "Local image generation error" in out


def test_analyze_image_exception_returns_error():
    out = _analyze_image("/tmp/definitely_nonexistent_file_xyz.png", "What is this?")
    assert "Vision analysis error" in out


def test_generate_image_hf_no_token_no_auth_header(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    call_kwargs = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        call_kwargs["headers"] = headers
        return MagicMock(status_code=200, content=b"bytes")

    fake_requests = MagicMock()
    fake_requests.post = fake_post
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _generate_image_hf("prompt", output_path="/tmp/x.png")
    assert "Authorization" not in call_kwargs.get("headers", {})


# ── synthesise_tool: dynamic flag and category ────────────────────────────────

@pytest.mark.asyncio
async def test_synthesise_tool_dynamic_flag_is_true(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=_valid_payload("dynamic_test"))]
    ))
    tool = await synthesise_tool("dynamic capability", client, reg)
    assert tool is not None
    assert tool.dynamic is True


@pytest.mark.asyncio
async def test_synthesise_tool_category_from_spec(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    payload = json.dumps({
        "name": "cat_tool",
        "description": "A categorised tool.",
        "category": "web",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "python_code": "def run(**kwargs):\n    return 'ok'",
    })
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=payload)]
    ))
    tool = await synthesise_tool("web capability", client, reg)
    assert tool is not None
    assert tool.category == "web"


# ── _make_fn: returns a callable ──────────────────────────────────────────────

def test_make_fn_returns_callable():
    code = "def run(**kwargs):\n    return 'callable'"
    fn = _make_fn(code)
    assert callable(fn)
    assert fn() == "callable"


# ── _make_fn: missing run function ────────────────────────────────────────────

def test_make_fn_missing_run_raises_key_error():
    code = "def not_run(**kwargs):\n    return 'nope'"
    with pytest.raises(KeyError):
        _make_fn(code)


# ── synthesise_tool: missing required key → returns None ─────────────────────

@pytest.mark.asyncio
async def test_synthesise_tool_missing_name_key_returns_none(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    bad_payload = '{"description": "no name key", "category": "general", "parameters": {}, "python_code": "def run(**k): return ok"}'
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=bad_payload)]
    ))
    tool = await synthesise_tool("missing name", client, reg)
    assert tool is None


@pytest.mark.asyncio
async def test_synthesise_tool_persists_file_to_tools_dir(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    from jarvis.tools.registry import ToolRegistry
    reg = ToolRegistry()
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=MagicMock(
        content=[MagicMock(text=_valid_payload("persisted_tool"))]
    ))
    tool = await synthesise_tool("persist test", client, reg)
    assert tool is not None
    assert (tmp_path / "persisted_tool.py").exists()


# ── _generate_image_hf: saves bytes to file ───────────────────────────────────

def test_generate_image_hf_saves_file_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr("jarvis.tools.image_gen.cfg.HF_API_TOKEN", "test-token")
    output_path = str(tmp_path / "out.png")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    fake_requests = MagicMock()
    fake_requests.post.return_value = fake_resp
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("a cat", output_path=output_path)
    assert Path(output_path).exists()
    assert "saved to" in out


# ── _describe_image: media_type mapping ──────────────────────────────────────

def test_describe_image_uses_jpeg_for_jpg(tmp_path, monkeypatch):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
    fake_client = MagicMock()
    fake_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="A photo")]
    )
    with patch("anthropic.Anthropic", return_value=fake_client):
        out = _describe_image(str(img))
    assert out == "A photo"
    call_kwargs = fake_client.messages.create.call_args.kwargs
    image_content = call_kwargs["messages"][0]["content"][0]
    assert image_content["source"]["media_type"] == "image/jpeg"
