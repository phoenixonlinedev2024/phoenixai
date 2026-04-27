"""Tests for jarvis.tools.github_tool and jarvis.tools.geo_weather_tools."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Inject fakes ──────────────────────────────────────────────────────────────

def _inject_fakes():
    for name in ("anthropic", "openai", "chromadb", "sentence_transformers", "modal"):
        sys.modules.setdefault(name, MagicMock())
    sys.modules["anthropic"].AsyncAnthropic = MagicMock
    # Fake PyGithub module
    fake_gh = MagicMock()
    sys.modules.setdefault("github", fake_gh)


_inject_fakes()

from jarvis.tools.geo_weather_tools import (  # noqa: E402
    _geocode,
    _get_weather,
    _log_analyse,
    _ocr_image,
    _read_rss,
)
from jarvis.tools.github_tool import (  # noqa: E402
    _comment_on_issue,
    _create_issue,
    _get_file_content,
    _gh_client,
    _list_issues,
    _list_prs,
    _list_repos,
    _search_code,
)


# ── _gh_client ────────────────────────────────────────────────────────────────

def test_gh_client_without_token(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "GITHUB_TOKEN", "")
    with pytest.raises(RuntimeError) as exc:
        _gh_client()
    assert "GITHUB_TOKEN" in str(exc.value)


def test_gh_client_with_token(monkeypatch):
    from jarvis.config import cfg
    monkeypatch.setattr(cfg, "GITHUB_TOKEN", "token123")
    fake_github_cls = MagicMock(return_value="client_instance")
    with patch.dict(sys.modules, {"github": MagicMock(Github=fake_github_cls)}):
        out = _gh_client()
    fake_github_cls.assert_called_once_with("token123")
    assert out == "client_instance"


# ── _list_repos ───────────────────────────────────────────────────────────────

def test_list_repos_user():
    repo = MagicMock(full_name="me/foo", description="A foo repo", stargazers_count=5)
    user = MagicMock()
    user.get_repos = MagicMock(return_value=[repo])
    gh = MagicMock()
    gh.get_user = MagicMock(return_value=user)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_repos()
    assert "me/foo" in out
    assert "5" in out


def test_list_repos_org():
    repo = MagicMock(full_name="acme/bar", description=None, stargazers_count=0)
    org = MagicMock()
    org.get_repos = MagicMock(return_value=[repo])
    gh = MagicMock()
    gh.get_organization = MagicMock(return_value=org)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_repos(org="acme")
    assert "acme/bar" in out
    assert "no description" in out


def test_list_repos_empty():
    gh = MagicMock()
    gh.get_user = MagicMock(return_value=MagicMock(get_repos=MagicMock(return_value=[])))
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_repos()
    assert "No repos" in out


def test_list_repos_error():
    with patch("jarvis.tools.github_tool._gh_client",
               side_effect=RuntimeError("auth failed")):
        out = _list_repos()
    assert "GitHub error" in out


# ── _list_issues ──────────────────────────────────────────────────────────────

def test_list_issues_returns_lines():
    issue = MagicMock(number=42, state="open", title="Bug", user=MagicMock(login="alice"))
    repo = MagicMock()
    repo.get_issues = MagicMock(return_value=[issue])
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_issues("me/foo")
    assert "#42" in out
    assert "Bug" in out
    assert "alice" in out


def test_list_issues_empty():
    repo = MagicMock()
    repo.get_issues = MagicMock(return_value=[])
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_issues("me/foo")
    assert "No issues" in out


def test_list_issues_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _list_issues("me/foo")
    assert "GitHub error" in out


# ── _create_issue ─────────────────────────────────────────────────────────────

def test_create_issue_success():
    issue = MagicMock(number=99, html_url="https://github.com/me/foo/issues/99")
    repo = MagicMock()
    repo.create_issue = MagicMock(return_value=issue)
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _create_issue("me/foo", "Title", "Body")
    assert "#99" in out
    assert "https://github.com/me/foo/issues/99" in out
    repo.create_issue.assert_called_once_with(title="Title", body="Body", labels=[])


def test_create_issue_with_labels():
    issue = MagicMock(number=1, html_url="u")
    repo = MagicMock()
    repo.create_issue = MagicMock(return_value=issue)
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        _create_issue("me/foo", "T", "B", labels=["bug", "urgent"])
    kwargs = repo.create_issue.call_args.kwargs
    assert kwargs["labels"] == ["bug", "urgent"]


def test_create_issue_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _create_issue("me/foo", "t", "b")
    assert "GitHub error" in out


# ── _list_prs ─────────────────────────────────────────────────────────────────

def test_list_prs_returns_lines():
    pr = MagicMock(number=7, state="open", title="Feature", user=MagicMock(login="bob"))
    repo = MagicMock()
    repo.get_pulls = MagicMock(return_value=[pr])
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_prs("me/foo")
    assert "#7" in out
    assert "Feature" in out
    assert "bob" in out


def test_list_prs_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _list_prs("me/foo")
    assert "GitHub error" in out


# ── _comment_on_issue ─────────────────────────────────────────────────────────

def test_comment_on_issue_success():
    c = MagicMock(html_url="https://example.com/c/1")
    issue = MagicMock()
    issue.create_comment = MagicMock(return_value=c)
    repo = MagicMock()
    repo.get_issue = MagicMock(return_value=issue)
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _comment_on_issue("me/foo", 10, "LGTM")
    assert "Comment posted" in out
    assert "https://example.com/c/1" in out
    issue.create_comment.assert_called_once_with("LGTM")


def test_comment_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _comment_on_issue("me/foo", 1, "x")
    assert "GitHub error" in out


# ── _get_file_content ─────────────────────────────────────────────────────────

def test_get_file_content_decodes():
    content = MagicMock()
    content.decoded_content = b"hello world"
    repo = MagicMock()
    repo.get_contents = MagicMock(return_value=content)
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _get_file_content("me/foo", "README.md", branch="dev")
    assert out == "hello world"
    repo.get_contents.assert_called_once_with("README.md", ref="dev")


def test_get_file_content_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _get_file_content("me/foo", "p")
    assert "GitHub error" in out


# ── _search_code ──────────────────────────────────────────────────────────────

def test_search_code_success():
    result = MagicMock(path="src/app.py")
    result.repository = MagicMock(full_name="me/foo")
    gh = MagicMock()
    gh.search_code = MagicMock(return_value=[result])
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _search_code("TODO")
    assert "me/foo/src/app.py" in out
    gh.search_code.assert_called_once_with("TODO")


def test_search_code_with_repo_filter():
    gh = MagicMock()
    gh.search_code = MagicMock(return_value=[])
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        _search_code("bug", repo="me/foo")
    gh.search_code.assert_called_once_with("bug repo:me/foo")


def test_search_code_error():
    with patch("jarvis.tools.github_tool._gh_client", side_effect=RuntimeError("x")):
        out = _search_code("q")
    assert "search error" in out.lower()


# ── github register_tools ─────────────────────────────────────────────────────

def test_github_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.github_tool import register_tools
    registry = build_registry()
    register_tools(registry)
    for name in ("github_list_repos", "github_list_issues", "github_create_issue",
                 "github_list_prs", "github_comment", "github_get_file", "github_search_code"):
        assert registry.get(name) is not None


# ── _geocode ──────────────────────────────────────────────────────────────────

def test_geocode_returns_results():
    fake_requests = MagicMock()
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(return_value=[
        {"display_name": "London, UK", "lat": "51.5", "lon": "-0.1"},
    ])
    fake_requests.get = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _geocode("London")
    assert "London, UK" in out
    assert "51.5" in out


def test_geocode_empty():
    fake_requests = MagicMock()
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(return_value=[])
    fake_requests.get = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _geocode("Atlantis")
    assert "No results" in out


def test_geocode_error():
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(side_effect=RuntimeError("network down"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _geocode("x")
    assert "Geocode error" in out


# ── _get_weather ──────────────────────────────────────────────────────────────

def test_get_weather_returns_forecast():
    fake_requests = MagicMock()
    geo_resp = MagicMock()
    geo_resp.json = MagicMock(return_value=[
        {"display_name": "Paris, FR", "lat": "48.8", "lon": "2.3"},
    ])
    weather_resp = MagicMock()
    weather_resp.json = MagicMock(return_value={
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_max": [10, 12],
            "temperature_2m_min": [4, 5],
            "precipitation_sum": [0.0, 1.5],
        }
    })
    fake_requests.get = MagicMock(side_effect=[geo_resp, weather_resp])
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _get_weather("Paris", days=2)
    assert "Paris, FR" in out
    assert "2024-01-01" in out
    assert "10" in out


def test_get_weather_location_not_found():
    fake_requests = MagicMock()
    geo_resp = MagicMock()
    geo_resp.json = MagicMock(return_value=[])
    fake_requests.get = MagicMock(return_value=geo_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _get_weather("Xanadu")
    assert "not found" in out.lower()


def test_get_weather_api_error():
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(side_effect=RuntimeError("api down"))
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _get_weather("x")
    assert "Weather error" in out


def test_get_weather_empty_daily():
    """Line 43: returns 'Weather data unavailable.' when daily dict is empty."""
    fake_requests = MagicMock()
    geo_resp = MagicMock()
    geo_resp.json = MagicMock(return_value=[
        {"display_name": "Test City", "lat": "10.0", "lon": "20.0"},
    ])
    weather_resp = MagicMock()
    weather_resp.json = MagicMock(return_value={"daily": {}})
    fake_requests.get = MagicMock(side_effect=[geo_resp, weather_resp])
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _get_weather("Test City")
    assert out == "Weather data unavailable."


# ── _read_rss ─────────────────────────────────────────────────────────────────

def test_read_rss_with_feedparser():
    fake_feedparser = MagicMock()
    fake_feed = MagicMock()
    fake_feed.entries = [
        {"title": "Post A", "link": "http://a", "summary": "first post summary"},
        {"title": "Post B", "link": "http://b", "summary": "second"},
    ]
    fake_feedparser.parse = MagicMock(return_value=fake_feed)
    with patch.dict(sys.modules, {"feedparser": fake_feedparser}):
        out = _read_rss("http://example.com/feed")
    assert "Post A" in out
    assert "Post B" in out


def test_read_rss_empty():
    fake_feedparser = MagicMock()
    fake_feed = MagicMock()
    fake_feed.entries = []
    fake_feedparser.parse = MagicMock(return_value=fake_feed)
    with patch.dict(sys.modules, {"feedparser": fake_feedparser}):
        out = _read_rss("http://nothing")
    assert "No entries" in out


def test_read_rss_feedparser_missing_fallback_success():
    """Lines 66-78: fallback XML parsing when feedparser is not installed."""
    rss_xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        '<title>Test</title>'
        '<item><title>Article One</title><link>http://example.com/1</link></item>'
        '</channel></rss>'
    )
    fake_requests = MagicMock()
    fake_resp = MagicMock()
    fake_resp.text = rss_xml
    fake_requests.get = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"feedparser": None, "requests": fake_requests}):
        out = _read_rss("http://example.com/feed")
    assert "Article One" in out


def test_read_rss_feedparser_missing_fallback_error():
    """Lines 79-80: fallback exception path when requests also fails."""
    fake_requests = MagicMock()
    fake_requests.get = MagicMock(side_effect=RuntimeError("network down"))
    with patch.dict(sys.modules, {"feedparser": None, "requests": fake_requests}):
        out = _read_rss("http://example.com/feed")
    assert "RSS error" in out


def test_read_rss_feedparser_parse_exception():
    """Lines 81-82: outer except catches errors from feedparser.parse() itself."""
    fake_feedparser = MagicMock()
    fake_feedparser.parse = MagicMock(side_effect=RuntimeError("feedparser crashed"))
    with patch.dict(sys.modules, {"feedparser": fake_feedparser}):
        out = _read_rss("http://bad.example")
    assert "RSS error" in out


# ── _ocr_image ────────────────────────────────────────────────────────────────

def test_ocr_image_missing_deps():
    with patch.dict(sys.modules, {"pytesseract": None, "PIL": None}):
        out = _ocr_image("/tmp/x.png")
    assert "not installed" in out.lower()


def test_ocr_image_error():
    fake_pytess = MagicMock()
    fake_pytess.image_to_string = MagicMock(side_effect=RuntimeError("bad image"))
    fake_pil = MagicMock()
    fake_pil.Image.open = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"pytesseract": fake_pytess, "PIL": fake_pil}):
        out = _ocr_image("/tmp/x.png")
    assert "OCR error" in out


# ── _log_analyse ──────────────────────────────────────────────────────────────

def test_log_analyse_counts_errors(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "INFO starting\n"
        "ERROR something broke\n"
        "DEBUG ok\n"
        "CRITICAL fatal failure\n"
        "INFO done\n"
    )
    out = _log_analyse(str(log))
    assert "Total lines: 5" in out
    assert "Errors: 2" in out


def test_log_analyse_with_pattern(tmp_path):
    log = tmp_path / "app.log"
    log.write_text("one\ntwo\nthree\nfour\n")
    out = _log_analyse(str(log), pattern="tw")
    assert "Total lines: 1" in out
    assert "two" in out


def test_log_analyse_tail_limit(tmp_path):
    log = tmp_path / "big.log"
    log.write_text("\n".join(f"line{i}" for i in range(1000)) + "\n")
    out = _log_analyse(str(log), tail=20)
    assert "Total lines: 20" in out


def test_log_analyse_missing_file():
    out = _log_analyse("/nonexistent/log.txt")
    assert "Log error" in out


# ── geo_weather register_tools ────────────────────────────────────────────────

def test_geo_weather_register_tools_populates_registry():
    from jarvis.tools.registry import build_registry
    from jarvis.tools.geo_weather_tools import register_tools
    registry = build_registry()
    register_tools(registry)
    for name in ("geocode", "get_weather", "read_rss", "ocr_image", "analyse_logs"):
        assert registry.get(name) is not None


# ── _log_analyse: case-insensitive error detection ────────────────────────────

def test_log_analyse_case_insensitive_error_detection(tmp_path):
    log = tmp_path / "mixed.log"
    log.write_text("ERROR fatal crash\nWARNING mild issue\nEXCEPTION null pointer\n")
    out = _log_analyse(str(log))
    assert "Errors: 2" in out  # ERROR + EXCEPTION counted


def test_log_analyse_no_errors(tmp_path):
    log = tmp_path / "clean.log"
    log.write_text("INFO all good\nDEBUG processing\nINFO done\n")
    out = _log_analyse(str(log))
    assert "Errors: 0" in out


def test_log_analyse_empty_file(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")
    out = _log_analyse(str(log))
    assert "Total lines: 0" in out
    assert "Errors: 0" in out


def test_log_analyse_regex_pattern_no_match(tmp_path):
    log = tmp_path / "miss.log"
    log.write_text("line1\nline2\nline3\n")
    out = _log_analyse(str(log), pattern="ZZZMATCH")
    assert "Total lines: 0" in out


# ── _geocode: aiohttp unavailable ────────────────────────────────────────────

def test_geocode_missing_requests(monkeypatch):
    with patch.dict(sys.modules, {"requests": None}):
        out = _geocode("Paris")
    assert "error" in out.lower() or "not installed" in out.lower() or "Geocode error" in out


# ── _read_rss: lxml/feedparser unavailable ────────────────────────────────────

def test_read_rss_missing_feedparser():
    with patch.dict(sys.modules, {"feedparser": None}):
        out = _read_rss("http://example.com/rss")
    assert "not installed" in out.lower() or "error" in out.lower()


# ── _ocr_image: success path with custom lang ─────────────────────────────────

def test_ocr_image_success_with_lang():
    fake_pytess = MagicMock()
    fake_pytess.image_to_string = MagicMock(return_value="extracted text")
    fake_img = MagicMock()
    fake_pil = MagicMock()
    fake_pil.Image.open = MagicMock(return_value=fake_img)
    with patch.dict(sys.modules, {"pytesseract": fake_pytess, "PIL": fake_pil}):
        out = _ocr_image("/tmp/test.png", lang="fra")
    assert out == "extracted text"
    fake_pytess.image_to_string.assert_called_once_with(fake_img, lang="fra")


# ── _log_analyse: fatal and critical keyword detection ────────────────────────

def test_log_analyse_counts_fatal_and_critical(tmp_path):
    log = tmp_path / "sev.log"
    log.write_text("FATAL system shutdown\nCRITICAL disk full\nINFO running\n")
    out = _log_analyse(str(log))
    assert "Errors: 2" in out


def test_log_analyse_output_includes_log_content(tmp_path):
    log = tmp_path / "out.log"
    log.write_text("INFO line one\nINFO line two\n")
    out = _log_analyse(str(log))
    assert "line one" in out


# ── _geocode: success returns multiple results ────────────────────────────────

def test_geocode_returns_multiple_results():
    fake_requests = MagicMock()
    fake_resp = MagicMock()
    fake_resp.json = MagicMock(return_value=[
        {"display_name": "Paris, France", "lat": "48.8566", "lon": "2.3522"},
        {"display_name": "Paris, Texas, USA", "lat": "33.66", "lon": "-95.55"},
    ])
    fake_requests.get = MagicMock(return_value=fake_resp)
    with patch.dict(sys.modules, {"requests": fake_requests}):
        out = _geocode("Paris")
    assert "Paris, France" in out
    assert "48.8566" in out


# ── _list_prs: empty result ───────────────────────────────────────────────────

def test_list_prs_empty():
    repo = MagicMock()
    repo.get_pulls = MagicMock(return_value=[])
    gh = MagicMock()
    gh.get_repo = MagicMock(return_value=repo)
    with patch("jarvis.tools.github_tool._gh_client", return_value=gh):
        out = _list_prs("me/foo")
    assert "No PRs" in out
