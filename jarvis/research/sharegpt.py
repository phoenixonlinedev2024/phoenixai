"""ShareGPT exporter — converts trajectories to ShareGPT format for fine-tuning.
Also supports trajectory compression for token efficiency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.research.trajectory import Trajectory, TrajectoryCollector


def to_sharegpt(trajectory: "Trajectory") -> dict:
    """Convert a trajectory to ShareGPT conversation format."""
    conversations = []
    for turn in trajectory.turns:
        if turn.role == "user":
            conversations.append({"from": "human", "value": turn.content})
        elif turn.role == "assistant":
            content = turn.content
            if turn.tool_calls:
                tool_str = "\n".join(
                    f"[TOOL: {tc['name']}({json.dumps(tc.get('input', {}))})]\n{tr.get('content', '')}"
                    for tc, tr in zip(turn.tool_calls, turn.tool_results or [{}] * len(turn.tool_calls))
                )
                content = f"{content}\n\n{tool_str}" if content else tool_str
            conversations.append({"from": "gpt", "value": content})

    return {
        "id": trajectory.id,
        "conversations": conversations,
        "task": trajectory.task,
        "outcome": trajectory.outcome,
        "reward": trajectory.reward,
    }


def compress_trajectory(trajectory: "Trajectory", max_turns: int = 20) -> "Trajectory":
    """Compress a long trajectory by keeping first turn, last N turns, and highest-reward turns."""
    from jarvis.research.trajectory import Trajectory

    if len(trajectory.turns) <= max_turns:
        return trajectory

    # Keep first turn (task statement) + last (max_turns - 1) turns
    important = trajectory.turns[:1] + trajectory.turns[-(max_turns - 1):]
    compressed = Trajectory(
        id=trajectory.id,
        session_id=trajectory.session_id,
        task=trajectory.task,
        outcome=trajectory.outcome,
        reward=trajectory.reward,
        metadata={**trajectory.metadata, "compressed": True, "original_turns": len(trajectory.turns)},
        timestamp=trajectory.timestamp,
    )
    compressed.turns = important
    return compressed


def export_dataset(
    collector: "TrajectoryCollector",
    output_path: str = "jarvis_dataset.jsonl",
    min_reward: float = 0.5,
    compress: bool = True,
    max_turns: int = 20,
) -> str:
    """Export all successful trajectories to a JSONL ShareGPT dataset file."""
    trajectories = collector.load_all()
    filtered = [t for t in trajectories if t.reward >= min_reward]

    if compress:
        filtered = [compress_trajectory(t, max_turns=max_turns) for t in filtered]

    records = [to_sharegpt(t) for t in filtered]

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return (
        f"Exported {len(records)} trajectories to {output_path}\n"
        f"(from {len(trajectories)} total, min_reward={min_reward}, compressed={compress})"
    )


def export_atropos_format(
    collector: "TrajectoryCollector",
    output_path: str = "jarvis_atropos.jsonl",
) -> str:
    """Export trajectories in Atropos RL training format."""
    trajectories = collector.load_all()
    records = []

    for traj in trajectories:
        messages = []
        for turn in traj.turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
                "reward": turn.reward if turn.reward is not None else 0.0,
            })
        records.append({
            "trajectory_id": traj.id,
            "task": traj.task,
            "messages": messages,
            "final_reward": traj.reward,
            "outcome": traj.outcome,
        })

    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return f"Exported {len(records)} trajectories to Atropos format: {output_path}"
