"""cv-autoresearch package."""

from cv_autoresearch.task import TaskDef

__all__ = ["TaskDef", "run"]


def __getattr__(name: str):
    """Keep legacy ``cv_autoresearch.run`` available without eager search imports."""
    if name == "run":
        from cv_autoresearch.autoresearch import run

        return run
    raise AttributeError(name)
