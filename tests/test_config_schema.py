"""Configuration schema for the unified dynamic-MoE FSCIL architecture."""
import unittest
from pathlib import Path
from main import setup_cfg


class ConfigSchemaTests(unittest.TestCase):
    def test_unified_configuration_has_only_active_mechanism_fields(self):
        root = Path(__file__).resolve().parents[1]
        cfg = setup_cfg(
            str(root / "configs/datasets/cub200.yaml"),
            str(root / "configs/trainers/bimc.yaml"),
        )
        self.assertTrue(cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE)
        self.assertEqual(cfg.TRAINER.BiMC.VISUAL_MOE.NUM_BASE_EXPERTS, 4)
        self.assertTrue(cfg.TRAINER.BiMC.VISION_CALIBRATION)
        self.assertTrue(cfg.TRAINER.BiMC.TEXT_CALIBRATION)
        self.assertTrue(hasattr(cfg.TRAINER.BiMC.PROTECTION, "OLD_GATE_WEIGHT"))
        self.assertFalse(hasattr(cfg.TRAINER.BiMC.VISUAL_MOE, "BASE_TOP_K"))
        self.assertFalse(hasattr(cfg.TRAINER.BiMC.VISUAL_MOE, "DESCRIPTOR_DIM"))


if __name__ == "__main__":
    unittest.main()
