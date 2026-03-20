"""Tests for the Claude metric config generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cv_autoresearch.advisor.metric_generator import (
    _build_metric_prompt,
    _call_claude,
    _parse_yaml_from_response,
    generate_metric_config,
)


class TestBuildMetricPrompt:
    def test_contains_task_description(self) -> None:
        prompt = _build_metric_prompt("object detection on COCO", "mAP")
        assert "object detection on COCO" in prompt

    def test_contains_primary_metric(self) -> None:
        prompt = _build_metric_prompt("image classification", "Accuracy")
        assert "Accuracy" in prompt

    def test_contains_both_fields(self) -> None:
        task = "semantic segmentation"
        metric = "IoU"
        prompt = _build_metric_prompt(task, metric)
        assert task in prompt
        assert metric in prompt


class TestParseYamlFromResponse:
    @pytest.mark.parametrize(
        "response,expected",
        [
            (
                "```yaml\n_target_: torchmetrics.Accuracy\ntask: multiclass\nnum_classes: 10\n```",
                {"_target_": "torchmetrics.Accuracy", "task": "multiclass", "num_classes": 10},
            ),
            (
                "```\n_target_: torchmetrics.Accuracy\ntask: binary\n```",
                {"_target_": "torchmetrics.Accuracy", "task": "binary"},
            ),
            (
                "_target_: torchmetrics.Accuracy\ntask: multiclass\nnum_classes: 5",
                {"_target_": "torchmetrics.Accuracy", "task": "multiclass", "num_classes": 5},
            ),
        ],
    )
    def test_parses_correctly(self, response: str, expected: dict) -> None:
        result = _parse_yaml_from_response(response)
        assert result == expected

    def test_strips_yaml_fences(self) -> None:
        response = "```yaml\n_target_: torchmetrics.Accuracy\n```"
        result = _parse_yaml_from_response(response)
        assert result == {"_target_": "torchmetrics.Accuracy"}

    def test_strips_plain_fences(self) -> None:
        response = "```\n_target_: torchmetrics.F1Score\n```"
        result = _parse_yaml_from_response(response)
        assert result == {"_target_": "torchmetrics.F1Score"}

    def test_handles_plain_yaml(self) -> None:
        response = "_target_: torchmetrics.MeanSquaredError"
        result = _parse_yaml_from_response(response)
        assert result == {"_target_": "torchmetrics.MeanSquaredError"}


class TestCallClaude:
    def test_returns_stdout_on_success(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "result"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            output = _call_claude("some prompt")

        assert output == "result"
        mock_run.assert_called_once()

    def test_raises_runtime_error_on_failure(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error msg"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="error msg"):
                _call_claude("some prompt")

    def test_passes_correct_command(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _call_claude("my prompt")

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        assert cmd[2] == "my prompt"


class TestGenerateMetricConfig:
    FAKE_YAML_RESPONSE = "_target_: torchmetrics.Accuracy\ntask: multiclass\nnum_classes: 10"

    def test_returns_dictconfig_with_target_key(self, tmp_path: pytest.TempPathFactory) -> None:
        output_path = str(tmp_path / "metric.yaml")

        with patch(
            "cv_autoresearch.advisor.metric_generator._call_claude",
            return_value=self.FAKE_YAML_RESPONSE,
        ):
            cfg = generate_metric_config(
                task_description="image classification",
                primary_metric="Accuracy",
                output_path=output_path,
            )

        assert "_target_" in cfg
        assert cfg["_target_"] == "torchmetrics.Accuracy"

    def test_writes_yaml_file(self, tmp_path: pytest.TempPathFactory) -> None:
        output_path = tmp_path / "subdir" / "metric.yaml"

        with patch(
            "cv_autoresearch.advisor.metric_generator._call_claude",
            return_value=self.FAKE_YAML_RESPONSE,
        ):
            generate_metric_config(
                task_description="image classification",
                primary_metric="Accuracy",
                output_path=str(output_path),
            )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_config_values_match_response(self, tmp_path: pytest.TempPathFactory) -> None:
        output_path = str(tmp_path / "metric.yaml")

        with patch(
            "cv_autoresearch.advisor.metric_generator._call_claude",
            return_value=self.FAKE_YAML_RESPONSE,
        ):
            cfg = generate_metric_config(
                task_description="image classification",
                primary_metric="Accuracy",
                output_path=output_path,
            )

        assert cfg["task"] == "multiclass"
        assert cfg["num_classes"] == 10
