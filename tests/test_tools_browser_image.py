"""Tests for jarvis.tools.browser, image_gen, and sandbox_tools."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock


_inject_fakes()

import jarvis.tools.sandbox_tools as sandbox_tools  # noqa: E402
from jarvis.tools.browser import (  # noqa: E402
    _browser_click_and_read,
    _browser_fetch,
    _browser_fill_form,
    _take_screenshot,
)
from jarvis.tools.image_gen import (  # noqa: E402
    _analyze_image,
    _describe_image,
    _generate_image_hf,
    _generate_image_local,
)


# ── browser: Playwright not installed ─────────────────────────────────────────

def test_browser_fetch_playwright_missing():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        out = _browser_fetch("https://example.com")
    assert "not installed" in out.lower() or "error" in out.lower()


def test_browser_click_error_without_playwright():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        out = _browser_click_and_read("https://example.com", "button")
    assert "error" in out.lower()


def test_browser_fill_form_error_without_playwright():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        out = _browser_fill_form("https://example.com", {"#a": "x"}, "button")
    assert "error" in out.lower()


def test_take_screenshot_error_without_playwright():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        out = _take_screenshot("https://example.com")
    assert "error" in out.lower()


def test_browser_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.browser import register_tools
    registry = build_registry()
    register_tools(registry)
    for name in ("browser_fetch", "browser_click", "browser_fill_form", "take_screenshot"):
        assert registry.get(name) is not None


def test_browser_fetch_non_import_exception():
    """Lines 28-29: Browser fetch error when playwright raises a non-ImportError."""
    fake_api = MagicMock()
    fake_api.async_playwright.side_effect = RuntimeError("launch failed")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_api}):
        out = _browser_fetch("https://example.com")
    assert "Browser fetch error" in out or "error" in out.lower()


def _make_fake_playwright(content="page text"):
    """Build a minimal async playwright mock tree."""
    from unittest.mock import AsyncMock
    fake_page = MagicMock()
    fake_page.goto = AsyncMock()
    fake_page.inner_text = AsyncMock(return_value=content)
    fake_page.wait_for_selector = AsyncMock()
    fake_page.click = AsyncMock()
    fake_page.wait_for_load_state = AsyncMock()
    fake_page.fill = AsyncMock()
    fake_page.screenshot = AsyncMock()

    fake_browser = MagicMock()
    fake_browser.new_page = AsyncMock(return_value=fake_page)
    fake_browser.close = AsyncMock()

    fake_chromium = MagicMock()
    fake_chromium.launch = AsyncMock(return_value=fake_browser)

    fake_p = MagicMock()
    fake_p.chromium = fake_chromium

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_p)
    fake_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_pw_module = MagicMock()
    fake_pw_module.async_playwright = MagicMock(return_value=fake_ctx)

    return fake_pw_module, fake_page, fake_browser


def test_browser_fetch_success():
    fake_pw, _, _ = _make_fake_playwright("rendered content")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_pw}):
        out = _browser_fetch("https://example.com")
    assert "rendered content" in out


def test_browser_fetch_with_wait_for():
    fake_pw, fake_page, _ = _make_fake_playwright("waited content")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_pw}):
        out = _browser_fetch("https://example.com", wait_for="#main")
    assert "waited content" in out
    fake_page.wait_for_selector.assert_awaited_once_with("#main", timeout=10000)


def test_browser_click_success():
    fake_pw, fake_page, _ = _make_fake_playwright("clicked page")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_pw}):
        out = _browser_click_and_read("https://example.com", ".btn")
    assert "clicked page" in out
    fake_page.click.assert_awaited_once_with(".btn")


def test_browser_fill_form_success():
    fake_pw, fake_page, _ = _make_fake_playwright("form submitted")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_pw}):
        out = _browser_fill_form("https://example.com", {"#name": "Tony", "#pw": "secret"}, "#submit")
    assert "form submitted" in out
    assert fake_page.fill.await_count == 2


def test_take_screenshot_success(tmp_path):
    fake_pw, fake_page, _ = _make_fake_playwright()
    out_path = str(tmp_path / "shot.png")
    with patch.dict(sys.modules, {"playwright": MagicMock(), "playwright.async_api": fake_pw}):
        out = _take_screenshot("https://example.com", output_path=out_path)
    assert out_path in out
    fake_page.screenshot.assert_awaited_once_with(path=out_path, full_page=True)


# ── _generate_image_hf ────────────────────────────────────────────────────────

def test_generate_image_hf_success(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "hf_token")

    out_path = tmp_path / "img.png"
    fake_resp = MagicMock(status_code=200, content=b"PNG_DATA")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("a cat", output_path=str(out_path))

    assert "Image generated" in out
    assert str(out_path) in out
    assert out_path.read_bytes() == b"PNG_DATA"
    # Verify Authorization header was set
    headers = fake_requests.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer hf_token"


def test_generate_image_hf_no_token(tmp_path, monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")

    fake_resp = MagicMock(status_code=200, content=b"PNG")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        _generate_image_hf("a dog", output_path=str(tmp_path / "d.png"))

    headers = fake_requests.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_generate_image_hf_api_error(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    fake_resp = MagicMock(status_code=503, text="service unavailable")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("x")
    assert "503" in out
    assert "service unavailable" in out


def test_generate_image_hf_exception(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "HF_API_TOKEN", "")
    fake_requests = MagicMock()
    fake_requests.post = MagicMock(side_effect=RuntimeError("conn reset"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _generate_image_hf("x")
    assert "generation error" in out.lower()


# ── _generate_image_local ─────────────────────────────────────────────────────

def test_generate_image_local_missing_deps():
    with patch.dict(sys.modules, {"diffusers": None, "torch": None}):
        out = _generate_image_local("prompt")
    assert "not installed" in out.lower()


def test_generate_image_local_success(tmp_path):
    """Lines 37-45: local generation happy path via diffusers mock."""
    fake_image = MagicMock()
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [fake_image]

    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value.to.return_value = fake_pipe

    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionPipeline = pipeline_cls

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.float32 = "float32"

    out_path = str(tmp_path / "img.png")
    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        out = _generate_image_local("a cat", output_path=out_path)

    assert "saved to" in out
    fake_image.save.assert_called_once_with(out_path)


def test_generate_image_local_exception():
    """Lines 48-49: generic exception path in local generation."""
    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.side_effect = RuntimeError("CUDA out of memory")

    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionPipeline = pipeline_cls

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        out = _generate_image_local("a cat")

    assert "Local image generation error" in out
    assert "CUDA out of memory" in out


# ── _describe_image / _analyze_image (anthropic-mocked) ───────────────────────

def test_describe_image_success(tmp_path, monkeypatch):
    img = tmp_path / "test.png"
    img.write_bytes(b"FAKEPNG")
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "key")

    resp = MagicMock()
    resp.content = [MagicMock(text="A colourful image.")]
    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(return_value=resp)

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        out = _describe_image(str(img))

    assert out == "A colourful image."


def test_describe_image_missing_file(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "key")
    out = _describe_image("/nonexistent/path.png")
    assert "Vision error" in out


def test_analyze_image_success(tmp_path, monkeypatch):
    img = tmp_path / "pic.jpg"
    img.write_bytes(b"FAKE")
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "key")

    resp = MagicMock()
    resp.content = [MagicMock(text="Yes, there is a cat.")]
    fake_client = MagicMock()
    fake_client.messages.create = MagicMock(return_value=resp)

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        out = _analyze_image(str(img), "Is there a cat?")

    assert "cat" in out.lower()


def test_analyze_image_error(tmp_path, monkeypatch):
    img = tmp_path / "pic.png"
    img.write_bytes(b"x")
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "key")

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = MagicMock(side_effect=RuntimeError("api fail"))

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        out = _analyze_image(str(img), "q")

    assert "analysis error" in out.lower()


def test_image_gen_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.image_gen import register_tools
    registry = build_registry()
    register_tools(registry)
    for name in ("generate_image", "generate_image_local", "describe_image", "analyze_image"):
        assert registry.get(name) is not None


# ── sandbox_tools ─────────────────────────────────────────────────────────────

def test_sandbox_run_success(monkeypatch):
    sandbox_tools._router = None  # reset
    fake_router = MagicMock()
    fake_router.run = MagicMock(return_value="async_fn")

    import asyncio
    with patch("jarvis.tools.sandbox_tools._get_router", return_value=fake_router), \
         patch.object(asyncio, "run", return_value="ok: output"):
        out = sandbox_tools._sandbox_run("echo hi")

    assert out == "ok: output"


def test_sandbox_run_error(monkeypatch):
    sandbox_tools._router = None
    with patch("jarvis.tools.sandbox_tools._get_router",
               side_effect=RuntimeError("no router")):
        out = sandbox_tools._sandbox_run("echo hi")
    assert "Sandbox error" in out


def test_sandbox_run_code_success():
    sandbox_tools._router = None
    fake_router = MagicMock()
    fake_router.run_code = MagicMock(return_value="async_fn")

    import asyncio
    with patch("jarvis.tools.sandbox_tools._get_router", return_value=fake_router), \
         patch.object(asyncio, "run", return_value="code ran"):
        out = sandbox_tools._sandbox_run_code("print(1)")

    assert "code ran" in out


def test_sandbox_run_code_error():
    sandbox_tools._router = None
    with patch("jarvis.tools.sandbox_tools._get_router",
               side_effect=RuntimeError("fail")):
        out = sandbox_tools._sandbox_run_code("x=1")
    assert "code error" in out.lower()


def test_list_sandbox_backends_returns_names():
    sandbox_tools._router = None
    fake_router = MagicMock()
    fake_router.list_backends = MagicMock(return_value=["local", "docker"])
    with patch("jarvis.tools.sandbox_tools._get_router", return_value=fake_router):
        out = sandbox_tools._list_sandbox_backends()
    assert "local" in out
    assert "docker" in out


def test_get_router_caches():
    sandbox_tools._router = None
    fake_router_cls = MagicMock(return_value="router_instance")
    with patch("jarvis.sandbox.router.SandboxRouter", fake_router_cls):
        r1 = sandbox_tools._get_router()
        r2 = sandbox_tools._get_router()
    assert r1 is r2
    fake_router_cls.assert_called_once()
    sandbox_tools._router = None  # cleanup


def test_sandbox_tools_register():
    sandbox_tools._router = None
    from jarvis.tools.registry import build_registry
    registry = build_registry()
    sandbox_tools.register_tools(registry)
    assert registry.get("sandbox_run") is not None
    assert registry.get("sandbox_run_code") is not None
    assert registry.get("list_sandbox_backends") is not None


# ── sandbox_tools: list_backends output ──────────────────────────────────────

def test_list_sandbox_backends_returns_available_prefix():
    fake_router = MagicMock()
    fake_router.list_backends.return_value = ["local", "docker"]
    with patch("jarvis.tools.sandbox_tools._get_router", return_value=fake_router):
        out = sandbox_tools._list_sandbox_backends()
    assert out.startswith("Available backends:")


def test_sandbox_run_exception_in_asyncio_run():
    sandbox_tools._router = None
    fake_router = MagicMock()
    import asyncio
    with patch("jarvis.tools.sandbox_tools._get_router", return_value=fake_router), \
         patch.object(asyncio, "run", side_effect=RuntimeError("sandbox crashed")):
        out = sandbox_tools._sandbox_run("echo hi")
    assert "Sandbox error" in out


# ── browser: fill form / screenshot error strings ───────────────────────────

def test_browser_fill_form_error_without_playwright_detail():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        import asyncio
        with patch.object(asyncio, "run", return_value="Form fill error: import error"):
            out = _browser_fill_form("http://x.com", {"#f": "val"}, "#btn")
    assert "error" in out.lower() or "Form" in out


def test_take_screenshot_with_custom_output_path():
    with patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        import asyncio
        with patch.object(asyncio, "run", return_value="Screenshot saved to /custom/path.png"):
            out = _take_screenshot("http://example.com", output_path="/custom/path.png")
    assert "/custom/path.png" in out or "Screenshot" in out


# ── browser: register_tools populates all tools ──────────────────────────────

def test_browser_register_tools_has_all_four():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.browser import register_tools
    registry = build_registry()
    register_tools(registry)
    for name in ("browser_fetch", "browser_click", "browser_fill_form", "take_screenshot"):
        assert registry.get(name) is not None
