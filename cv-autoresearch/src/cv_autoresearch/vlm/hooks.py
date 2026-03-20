"""VLM hook infrastructure for future vision-language model integration."""

from __future__ import annotations

import asyncio
from typing import Any


class VLMHooks:
    """No-op base class for VLM integration hooks.

    Provides three hook points in the autoresearch loop:
    - analyze_iteration: called after each trial
    - analyze_phase: called after each search phase
    - analyze_experiment: called at the end of the full experiment

    Subclass and override to add real VLM analysis.
    """

    def analyze_iteration(
        self,
        trial_result: Any,
        history: Any,
        baseline: Any,
    ) -> None:
        """Called after each trial iteration. No-op in base class.

        Args:
            trial_result: IterationResult for the completed trial.
            history: Current SearchHistory.
            baseline: Current Baseline.
        """

    def analyze_phase(
        self,
        phase: Any,
        study: Any,
        baseline: Any,
    ) -> None:
        """Called after each search phase completes. No-op in base class.

        Args:
            phase: Phase identifier that just completed.
            study: Optuna Study for the phase.
            baseline: Current Baseline.
        """

    def analyze_experiment(
        self,
        history: Any,
        baseline: Any,
    ) -> None:
        """Called at the end of the full experiment. No-op in base class.

        Args:
            history: Final SearchHistory.
            baseline: Final Baseline.
        """


class AsyncVLMHooks(VLMHooks):
    """Fire-and-forget async VLM hooks.

    Schedules each hook as a coroutine on the event loop without awaiting.
    The actual hook implementations are left empty for future VLM integration.

    Uses asyncio.ensure_future for non-blocking hook dispatch.
    """

    def analyze_iteration(
        self,
        trial_result: Any,
        history: Any,
        baseline: Any,
    ) -> None:
        """Schedule iteration analysis coroutine non-blocking.

        Args:
            trial_result: IterationResult for the completed trial.
            history: Current SearchHistory.
            baseline: Current Baseline.
        """
        asyncio.ensure_future(self._analyze_iteration_async(trial_result, history, baseline))

    def analyze_phase(
        self,
        phase: Any,
        study: Any,
        baseline: Any,
    ) -> None:
        """Schedule phase analysis coroutine non-blocking.

        Args:
            phase: Phase identifier that just completed.
            study: Optuna Study for the phase.
            baseline: Current Baseline.
        """
        asyncio.ensure_future(self._analyze_phase_async(phase, study, baseline))

    def analyze_experiment(
        self,
        history: Any,
        baseline: Any,
    ) -> None:
        """Schedule experiment analysis coroutine non-blocking.

        Args:
            history: Final SearchHistory.
            baseline: Final Baseline.
        """
        asyncio.ensure_future(self._analyze_experiment_async(history, baseline))

    async def _analyze_iteration_async(
        self, trial_result: Any, history: Any, baseline: Any
    ) -> None:
        """Async iteration analysis stub. Override to add VLM analysis."""

    async def _analyze_phase_async(self, phase: Any, study: Any, baseline: Any) -> None:
        """Async phase analysis stub. Override to add VLM analysis."""

    async def _analyze_experiment_async(self, history: Any, baseline: Any) -> None:
        """Async experiment analysis stub. Override to add VLM analysis."""
