"""Agent-editable task wiring contract."""

from cv_autoresearch.engine.task_wiring.runtime import (
    AgentWiredTask,
    build_task,
    evaluate_after_training,
    pretrain_evaluate,
)

__all__ = [
    "AgentWiredTask",
    "build_task",
    "evaluate_after_training",
    "pretrain_evaluate",
]
