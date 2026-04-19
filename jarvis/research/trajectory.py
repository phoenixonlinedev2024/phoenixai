"""Trajectory collector — captures conversation trajectories for RL/fine-tuning."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from jarvis.config import cfg


@dataclass
class Turn:
    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    reward: float | None = None


@dataclass
class Trajectory:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    task: str = ""
    outcome: str = "unknown"  # success | failure | unknown
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_turn(self, role: str, content: str, **kwargs) -> None:
        self.turns.append(Turn(role=role, content=content, **kwargs))

    def set_outcome(self, outcome: str, reward: float = 0.0) -> None:
        self.outcome = outcome
        self.reward = reward
        # Backfill reward to last assistant turn
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                turn.reward = reward
                break

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "task": self.task,
            "outcome": self.outcome,
            "reward": self.reward,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tool_calls": t.tool_calls,
                    "tool_results": t.tool_results,
                    "reward": t.reward,
                }
                for t in self.turns
            ],
        }


class TrajectoryCollector:
    """Collects and persists conversation trajectories for research and fine-tuning."""

    def __init__(self) -> None:
        self._dir = cfg.DATA_DIR / "trajectories"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, Trajectory] = {}

    def start(self, session_id: str, task: str = "") -> Trajectory:
        traj = Trajectory(session_id=session_id, task=task)
        self._active[session_id] = traj
        return traj

    def get(self, session_id: str) -> Trajectory | None:
        return self._active.get(session_id)

    def record_turn(self, session_id: str, role: str, content: str, **kwargs) -> None:
        traj = self._active.get(session_id)
        if traj:
            traj.add_turn(role, content, **kwargs)

    def complete(self, session_id: str, outcome: str = "success", reward: float = 1.0) -> str:
        traj = self._active.pop(session_id, None)
        if not traj:
            return "No active trajectory for this session."
        traj.set_outcome(outcome, reward)
        path = self._dir / f"{traj.id}.json"
        path.write_text(json.dumps(traj.to_dict(), indent=2), encoding="utf-8")
        return f"Trajectory {traj.id} saved ({len(traj.turns)} turns, reward={reward})."

    def load_all(self) -> list[Trajectory]:
        trajectories = []
        for p in self._dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                traj = Trajectory(
                    id=data["id"],
                    session_id=data.get("session_id", ""),
                    task=data.get("task", ""),
                    outcome=data.get("outcome", "unknown"),
                    reward=data.get("reward", 0.0),
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp", ""),
                )
                for t in data.get("turns", []):
                    traj.turns.append(Turn(**t))
                trajectories.append(traj)
            except Exception:
                pass
        return trajectories

    def stats(self) -> dict:
        all_traj = self.load_all()
        return {
            "total": len(all_traj),
            "success": sum(1 for t in all_traj if t.outcome == "success"),
            "failure": sum(1 for t in all_traj if t.outcome == "failure"),
            "avg_reward": sum(t.reward for t in all_traj) / max(len(all_traj), 1),
            "avg_turns": sum(len(t.turns) for t in all_traj) / max(len(all_traj), 1),
        }
