"""GitHub tools — manage repos, issues, PRs autonomously (PyGithub, free)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.config import cfg

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _gh_client():
    if not cfg.GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set in .env")
    from github import Github
    return Github(cfg.GITHUB_TOKEN)


def _list_repos(org: str = "") -> str:
    try:
        gh = _gh_client()
        if org:
            items = gh.get_organization(org).get_repos()
        else:
            items = gh.get_user().get_repos()
        lines = [f"{r.full_name} — {r.description or 'no description'} ({r.stargazers_count}★)" for r in list(items)[:20]]
        return "\n".join(lines) or "No repos found."
    except Exception as exc:
        return f"GitHub error: {exc}"


def _list_issues(repo: str, state: str = "open") -> str:
    try:
        gh = _gh_client()
        r = gh.get_repo(repo)
        issues = r.get_issues(state=state)
        lines = [f"#{i.number} [{i.state}] {i.title} — {i.user.login}" for i in list(issues)[:20]]
        return "\n".join(lines) or "No issues found."
    except Exception as exc:
        return f"GitHub error: {exc}"


def _create_issue(repo: str, title: str, body: str, labels: list | None = None) -> str:
    try:
        gh = _gh_client()
        r = gh.get_repo(repo)
        issue = r.create_issue(title=title, body=body, labels=labels or [])
        return f"Created issue #{issue.number}: {issue.html_url}"
    except Exception as exc:
        return f"GitHub error: {exc}"


def _list_prs(repo: str, state: str = "open") -> str:
    try:
        gh = _gh_client()
        r = gh.get_repo(repo)
        prs = r.get_pulls(state=state)
        lines = [f"#{p.number} [{p.state}] {p.title} — {p.user.login}" for p in list(prs)[:20]]
        return "\n".join(lines) or "No PRs found."
    except Exception as exc:
        return f"GitHub error: {exc}"


def _comment_on_issue(repo: str, issue_number: int, comment: str) -> str:
    try:
        gh = _gh_client()
        r = gh.get_repo(repo)
        issue = r.get_issue(issue_number)
        c = issue.create_comment(comment)
        return f"Comment posted: {c.html_url}"
    except Exception as exc:
        return f"GitHub error: {exc}"


def _get_file_content(repo: str, path: str, branch: str = "main") -> str:
    try:
        gh = _gh_client()
        r = gh.get_repo(repo)
        content = r.get_contents(path, ref=branch)
        return content.decoded_content.decode("utf-8")
    except Exception as exc:
        return f"GitHub error: {exc}"


def _search_code(query: str, repo: str = "") -> str:
    try:
        gh = _gh_client()
        q = f"{query} repo:{repo}" if repo else query
        results = gh.search_code(q)
        lines = [f"{r.repository.full_name}/{r.path}" for r in list(results)[:10]]
        return "\n".join(lines) or "No results."
    except Exception as exc:
        return f"GitHub search error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool

    registry.register(Tool(
        name="github_list_repos",
        description="List GitHub repositories for the authenticated user or an org.",
        input_schema={
            "type": "object",
            "properties": {"org": {"type": "string", "description": "Organisation name (optional)"}},
        },
        fn=_list_repos,
        category="api",
    ))

    registry.register(Tool(
        name="github_list_issues",
        description="List issues in a GitHub repo.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/repo"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
            },
            "required": ["repo"],
        },
        fn=_list_issues,
        category="api",
    ))

    registry.register(Tool(
        name="github_create_issue",
        description="Create a new GitHub issue.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["repo", "title", "body"],
        },
        fn=_create_issue,
        category="api",
    ))

    registry.register(Tool(
        name="github_list_prs",
        description="List pull requests in a GitHub repo.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
            },
            "required": ["repo"],
        },
        fn=_list_prs,
        category="api",
    ))

    registry.register(Tool(
        name="github_comment",
        description="Post a comment on a GitHub issue or PR.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "issue_number": {"type": "integer"},
                "comment": {"type": "string"},
            },
            "required": ["repo", "issue_number", "comment"],
        },
        fn=_comment_on_issue,
        category="api",
    ))

    registry.register(Tool(
        name="github_get_file",
        description="Get the content of a file in a GitHub repository.",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "branch": {"type": "string", "default": "main"},
            },
            "required": ["repo", "path"],
        },
        fn=_get_file_content,
        category="api",
    ))

    registry.register(Tool(
        name="github_search_code",
        description="Search code across GitHub repositories.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "repo": {"type": "string", "description": "Limit to owner/repo (optional)"},
            },
            "required": ["query"],
        },
        fn=_search_code,
        category="api",
    ))
