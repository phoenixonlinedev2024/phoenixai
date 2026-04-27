"""Browser automation tools — Playwright for JS-rendered pages and form interaction."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _browser_fetch(url: str, wait_for: str | None = None) -> str:
    """Fetch a fully JS-rendered page using Playwright."""
    async def _run():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                if wait_for:
                    await page.wait_for_selector(wait_for, timeout=10000)
                content = await page.inner_text("body")
                await browser.close()
                return content[:8000]
        except ImportError:
            return "Playwright not installed. Run: pip install playwright && playwright install chromium"
        except Exception as exc:
            return f"Browser fetch error: {exc}"
    return asyncio.run(_run())


def _browser_click_and_read(url: str, selector: str) -> str:
    """Navigate to a URL, click an element, return the resulting page text."""
    async def _run():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.click(selector)
                await page.wait_for_load_state("networkidle")
                content = await page.inner_text("body")
                await browser.close()
                return content[:8000]
        except Exception as exc:
            return f"Browser click error: {exc}"
    return asyncio.run(_run())


def _browser_fill_form(url: str, fields: dict, submit_selector: str) -> str:
    """Fill a web form and submit it. fields is {css_selector: value}."""
    async def _run():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                for selector, value in fields.items():
                    await page.fill(selector, str(value))
                await page.click(submit_selector)
                await page.wait_for_load_state("networkidle")
                content = await page.inner_text("body")
                await browser.close()
                return content[:4000]
        except Exception as exc:
            return f"Form fill error: {exc}"
    return asyncio.run(_run())


def _take_screenshot(url: str, output_path: str = "/tmp/jarvis_screenshot.png") -> str:
    """Take a screenshot of a URL and save it."""
    async def _run():
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.screenshot(path=output_path, full_page=True)
                await browser.close()
                return f"Screenshot saved to {output_path}"
        except Exception as exc:
            return f"Screenshot error: {exc}"
    return asyncio.run(_run())


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="browser_fetch",
        description="Fetch a fully JS-rendered web page (runs real Chromium). Use when web_fetch fails on dynamic sites.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "wait_for": {"type": "string", "description": "CSS selector to wait for before reading"},
            },
            "required": ["url"],
        },
        fn=_browser_fetch,
        category="web",
    ))

    registry.register(Tool(
        name="browser_click",
        description="Navigate to a URL, click a CSS element, return the resulting page.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "selector": {"type": "string", "description": "CSS selector to click"},
            },
            "required": ["url", "selector"],
        },
        fn=_browser_click_and_read,
        category="web",
    ))

    registry.register(Tool(
        name="browser_fill_form",
        description="Fill and submit a web form. fields is a dict of {CSS selector: value}.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "fields": {"type": "object", "description": "Dict of CSS selector to value"},
                "submit_selector": {"type": "string", "description": "CSS selector for submit button"},
            },
            "required": ["url", "fields", "submit_selector"],
        },
        fn=_browser_fill_form,
        category="web",
    ))

    registry.register(Tool(
        name="take_screenshot",
        description="Take a full-page screenshot of any URL and save it as a PNG.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "output_path": {"type": "string", "default": "/tmp/jarvis_screenshot.png"},
            },
            "required": ["url"],
        },
        fn=_take_screenshot,
        category="web",
    ))
