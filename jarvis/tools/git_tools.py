"""Git tools — full git workflow via gitpython."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.tools.registry import ToolRegistry


def _git_status(repo_path: str = ".") -> str:
    try:
        import git
        repo = git.Repo(repo_path)
        staged = [f.a_path for f in repo.index.diff("HEAD")]
        unstaged = [f.a_path for f in repo.index.diff(None)]
        untracked = repo.untracked_files
        return json.dumps({"branch": repo.active_branch.name, "staged": staged, "unstaged": unstaged, "untracked": untracked}, indent=2)
    except ImportError:
        return "gitpython not installed. Run: pip install gitpython"
    except Exception as exc:
        return f"Git status error: {exc}"


def _git_log(repo_path: str = ".", n: int = 10) -> str:
    try:
        import git
        repo = git.Repo(repo_path)
        commits = list(repo.iter_commits(max_count=n))
        lines = [f"{c.hexsha[:8]} {c.committed_datetime.strftime('%Y-%m-%d')} {c.author.name}: {c.message.strip()[:80]}" for c in commits]
        return "\n".join(lines)
    except Exception as exc:
        return f"Git log error: {exc}"


def _git_clone(url: str, dest: str = "", branch: str = "") -> str:
    try:
        import git
        kwargs = {}
        if branch:
            kwargs["branch"] = branch
        target = dest or url.rstrip("/").split("/")[-1].replace(".git", "")
        git.Repo.clone_from(url, target, **kwargs)
        return f"Cloned {url} to {target}"
    except Exception as exc:
        return f"Clone error: {exc}"


def _git_commit(repo_path: str, message: str, add_all: bool = True) -> str:
    try:
        import git
        repo = git.Repo(repo_path)
        if add_all:
            repo.git.add(A=True)
        repo.index.commit(message)
        return f"Committed: {message}"
    except Exception as exc:
        return f"Commit error: {exc}"


def _git_diff(repo_path: str = ".", staged: bool = False) -> str:
    try:
        import git
        repo = git.Repo(repo_path)
        diff = repo.index.diff("HEAD") if staged else repo.index.diff(None)
        parts = []
        for d in diff:
            parts.append(f"--- {d.a_path}\n+++ {d.b_path}")
        return "\n".join(parts)[:5000] or "No changes."
    except Exception as exc:
        return f"Diff error: {exc}"


def _git_branch(repo_path: str = ".", name: str = "", checkout: bool = False) -> str:
    try:
        import git
        repo = git.Repo(repo_path)
        if not name:
            branches = [b.name for b in repo.branches]
            return f"Branches: {', '.join(branches)} (current: {repo.active_branch.name})"
        if checkout:
            repo.git.checkout(b=name) if name not in [b.name for b in repo.branches] else repo.git.checkout(name)
            return f"Switched to branch: {name}"
        repo.create_head(name)
        return f"Created branch: {name}"
    except Exception as exc:
        return f"Branch error: {exc}"


def register_tools(registry: "ToolRegistry") -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(name="git_status", description="Get git status: branch, staged, unstaged, untracked files.",
        input_schema={"type":"object","properties":{"repo_path":{"type":"string","default":"."}}},
        fn=_git_status, category="code"))
    registry.register(Tool(name="git_log", description="Get recent git commit history.",
        input_schema={"type":"object","properties":{"repo_path":{"type":"string","default":"."},"n":{"type":"integer","default":10}}},
        fn=_git_log, category="code"))
    registry.register(Tool(name="git_clone", description="Clone a git repository.",
        input_schema={"type":"object","properties":{"url":{"type":"string"},"dest":{"type":"string"},"branch":{"type":"string"}},"required":["url"]},
        fn=_git_clone, category="code"))
    registry.register(Tool(name="git_commit", description="Stage all changes and commit with a message.",
        input_schema={"type":"object","properties":{"repo_path":{"type":"string"},"message":{"type":"string"},"add_all":{"type":"boolean","default":True}},"required":["repo_path","message"]},
        fn=_git_commit, category="code"))
    registry.register(Tool(name="git_diff", description="Show uncommitted changes in a repo.",
        input_schema={"type":"object","properties":{"repo_path":{"type":"string","default":"."},"staged":{"type":"boolean","default":False}}},
        fn=_git_diff, category="code"))
    registry.register(Tool(name="git_branch", description="List, create, or checkout git branches.",
        input_schema={"type":"object","properties":{"repo_path":{"type":"string","default":"."},"name":{"type":"string"},"checkout":{"type":"boolean","default":False}}},
        fn=_git_branch, category="code"))
