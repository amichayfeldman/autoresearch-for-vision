from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProgramContractTest(unittest.TestCase):
    def test_program_md_keeps_karpathy_shape(self) -> None:
        text = Path("program.md").read_text()

        self.assertIn("There is no separate manager system", text)
        self.assertIn("`train.py` is the program", text)
        self.assertIn("Do not create a hidden manager", text)
        self.assertIn("agents/cv-wiring-agent.md", text)
        self.assertIn("skills/cv-task-bridge/SKILL.md", text)
        self.assertIn("skills/cv-metric-bridge/SKILL.md", text)
        self.assertIn("PyTorch Lightning", text)
        self.assertIn("Albumentations", text)
        self.assertIn("library model", text)
        self.assertIn("External repo path", text)
        self.assertIn("Model import path", text)
        self.assertIn("TRAINING_BACKEND", text)
        self.assertIn("precision", text)
        self.assertIn("recall", text)

    def test_wiring_guidance_is_prompt_only_and_allows_research(self) -> None:
        agent = Path("agents/cv-wiring-agent.md").read_text()
        task_skill = Path("skills/cv-task-bridge/SKILL.md").read_text()
        metric_skill = Path("skills/cv-metric-bridge/SKILL.md").read_text()

        self.assertIn("Use this prompt only during Phase 1", agent)
        self.assertIn("There is no manager process", agent)
        self.assertIn("You may search the internet during wiring", agent)
        self.assertIn("PyTorch Lightning", agent)
        self.assertIn("Albumentations", agent)
        self.assertIn("preconfigured from a library or checkpoint", agent)
        self.assertIn("installed Python package", agent)
        self.assertIn("MODEL_IMPORT", agent)
        self.assertIn("TRAINING_BACKEND = \"external\"", agent)
        self.assertIn("model.train(...)", agent)
        self.assertIn("official dataset documentation", task_skill)
        self.assertIn("build_model()", task_skill)
        self.assertIn("external repo", task_skill)
        self.assertIn("numeric `precision` and `recall`", metric_skill)
        self.assertIn("true positives", metric_skill)

    def test_train_py_uses_cv_libraries_as_wiring_surface(self) -> None:
        text = Path("train.py").read_text()

        self.assertIn("def build_model", text)
        self.assertIn("def build_train_augmentation", text)
        self.assertIn("import lightning.pytorch as pl", text)
        self.assertIn("import albumentations as A", text)
        self.assertIn("MODEL_SOURCE", text)
        self.assertIn("EXTERNAL_REPO_PATH", text)
        self.assertIn("installed package or repo", text)
        self.assertIn("EXTERNAL_TRAIN_IMPORT", text)
        self.assertIn("def fit_with_external_source", text)
        self.assertIn("def collect_external_predictions", text)

    def test_external_backend_can_bypass_lightning_dependency(self) -> None:
        import train

        with patch.object(train, "TRAINING_BACKEND", "external"):
            self.assertNotIn("lightning", train.missing_runtime_packages())

    def test_train_run_writes_numeric_cv_metrics(self) -> None:
        import train

        missing = train.missing_runtime_packages()
        if missing:
            self.skipTest("missing runtime package(s): " + ", ".join(missing))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            with patch.object(train, "OUTPUT_DIR", run_dir):
                result = train.run()

            metrics = json.loads((run_dir / "metrics.json").read_text())
            pretrain_metrics = json.loads((run_dir / "pretrain_metrics.json").read_text())

            self.assertEqual(result.precision, metrics["precision"])
            self.assertIsInstance(metrics["precision"], float)
            self.assertIsInstance(metrics["recall"], float)
            self.assertIsInstance(pretrain_metrics["precision"], float)
            self.assertIsInstance(pretrain_metrics["recall"], float)
            self.assertTrue((run_dir / "checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
