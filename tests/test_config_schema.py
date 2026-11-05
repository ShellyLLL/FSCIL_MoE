"""Configuration-schema regression tests for the FSCIL entry point."""
import tempfile
import unittest
from pathlib import Path

from main import setup_cfg


class ConfigSchemaTests(unittest.TestCase):
    def test_bimc_config_accepts_registered_protection_fields(self):
        root = Path(__file__).resolve().parents[1]
        cfg = setup_cfg(
            str(root / "configs/datasets/cub200.yaml"),
            str(root / "configs/trainers/bimc.yaml"),
        )
        self.assertAlmostEqual(
            cfg.TRAINER.BiMC.PROTECTION.OLD_TASK_RECALL_TOLERANCE, 0.02
        )
        self.assertAlmostEqual(cfg.TRAINER.BiMC.PROTECTION.GATE_WEIGHT, 1.0)

    def test_legacy_bimc_config_inherits_protection_defaults(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy_bimc.yaml"
            legacy.write_text(
                "TRAINER:\n  BiMC:\n    PROTECTION:\n"
                "      ALIGN_WEIGHT: 0.1\n      GATE_WEIGHT: 1.0\n",
                encoding="utf-8",
            )
            cfg = setup_cfg(
                str(root / "configs/datasets/cub200.yaml"), str(legacy)
            )
            self.assertAlmostEqual(
                cfg.TRAINER.BiMC.PROTECTION.OLD_TASK_RECALL_TOLERANCE, 0.02
            )


if __name__ == "__main__":
    unittest.main()
