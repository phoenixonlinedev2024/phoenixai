"""Geo, weather, and RSS tools — all free APIs, no key needed."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _geocode(address: str) -> str:
    try:
        import requests
        resp = requests.get("https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 3},
            headers={"User-Agent": "JARVIS/1.0"}, timeout=10)
        results = resp.json()
        if not results:
            return f"No results for '{address}'"
        out = []
        for r in results:
            out.append(f"{r['display_name']} — lat:{r['lat']} lon:{r['lon']}")
        return "\n".join(out)
    except Exception as exc:
        return f"Geocode error: {exc}"


def _get_weather(location: str, days: int = 3) -> str:
    try:
        import requests
        # Geocode first
        geo = requests.get("https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "JARVIS/1.0"}, timeout=10).json()
        if not geo:
            return f"Location not found: {location}"
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        weather = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                    "forecast_days": days, "timezone": "auto"}, timeout=10).json()
        daily = weather.get("daily", {})
        if not daily:
            return "Weather data unavailable."
        lines = [f"Weather for {geo[0]['display_name']}:"]
        for i, date in enumerate(daily.get("time", [])[:days]):
            hi = daily["temperature_2m_max"][i]
            lo = daily["temperature_2m_min"][i]
            rain = daily["precipitation_sum"][i]
            lines.append(f"  {date}: {lo}°C – {hi}°C, rain: {rain}mm")
        return "\n".join(lines)
    except Exception as exc:
        return f"Weather error: {exc}"


def _read_rss(url: str, max_items: int = 10) -> str:
    try:
        import feedparser
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:max_items]:
            title = e.get("title", "No title")
            link = e.get("link", "")
            summary = e.get("summary", "")[:200]
            items.append(f"**{title}**\n{link}\n{summary}")
        return "\n\n".join(items) or "No entries found."
    except ImportError:
        try:
            import requests
            from xml.etree import ElementTree as ET
            resp = requests.get(url, timeout=15, headers={"User-Agent": "JARVIS/1.0"})
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = []
            for item in root.findall(".//item")[:max_items] or root.findall(".//atom:entry", ns)[:max_items]:
                title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "")
                link = (item.findtext("link") or "")
                items.append(f"**{title.strip()}** — {link.strip()}")
            return "\n".join(items) or "No items found."
        except Exception as exc2:
            return f"RSS error: {exc2}"
    except Exception as exc:
        return f"RSS error: {exc}"


def _ocr_image(image_path: str, lang: str = "eng") -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, lang=lang)
    except ImportError:
        return "pytesseract/Pillow not installed. Run: pip install pytesseract Pillow (also: apt install tesseract-ocr)"
    except Exception as exc:
        return f"OCR error: {exc}"


def _log_analyse(path: str, pattern: str = "", tail: int = 100) -> str:
    try:
        import re
        lines = open(path, encoding="utf-8", errors="replace").readlines()
        lines = lines[-tail:]
        if pattern:
            lines = [line for line in lines if re.search(pattern, line, re.IGNORECASE)]
        errors = [line for line in lines if any(w in line.lower() for w in ("error", "exception", "critical", "fatal"))]
        return f"Total lines: {len(lines)}, Errors: {len(errors)}\n\n" + "".join(lines[:50])
    except Exception as exc:
        return f"Log error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="geocode", description="Geocode an address or place name to lat/lon coordinates using OpenStreetMap (free).",
        input_schema={"type":"object","properties":{"address":{"type":"string"}},"required":["address"]},
        fn=_geocode, category="api"))
    registry.register(Tool(name="get_weather", description="Get weather forecast for any location for up to 7 days (free OpenMeteo API).",
        input_schema={"type":"object","properties":{"location":{"type":"string"},"days":{"type":"integer","default":3}},"required":["location"]},
        fn=_get_weather, category="api"))
    registry.register(Tool(name="read_rss", description="Fetch and parse an RSS or Atom feed. Returns titles, links, and summaries.",
        input_schema={"type":"object","properties":{"url":{"type":"string"},"max_items":{"type":"integer","default":10}},"required":["url"]},
        fn=_read_rss, category="web"))
    registry.register(Tool(name="ocr_image", description="Extract text from an image file using Tesseract OCR.",
        input_schema={"type":"object","properties":{"image_path":{"type":"string"},"lang":{"type":"string","default":"eng"}},"required":["image_path"]},
        fn=_ocr_image, category="media"))
    registry.register(Tool(name="analyse_logs", description="Analyse a log file: filter by pattern, count errors, show tail.",
        input_schema={"type":"object","properties":{"path":{"type":"string"},"pattern":{"type":"string"},"tail":{"type":"integer","default":100}},"required":["path"]},
        fn=_log_analyse, category="system"))
