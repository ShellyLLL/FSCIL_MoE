"""Regression tests for the unified AE-driven FSCIL MoE configuration."""

import unittest

from main import setup_cfg


class ConfigSchemaTests(unittest.TestCase):
    def test_cub200_uses_only_current_unified_moe_contract(self):
        cfg = setup_cfg("./configs/datasets/cub200.yaml", "./configs/trainers/bimc.yaml")
        moe = cfg.TRAINER.BiMC.VISUAL_MOE
        incremental = cfg.TRAINER.BiMC.INCREMENTAL

        self.assertEqual(moe.INCREMENTAL_GATE_BIAS_INIT, -4.0)
        self.assertEqual(moe.SEMANTIC_RETRIEVAL_TOP_K, 5)
        self.assertEqual(moe.FEATURE_SPACE_MIN_COSINE, 0.98)
        self.assertEqual(moe.FEATURE_SPACE_MAX_ACCURACY_DROP, 0.0)
        self.assertEqual(incremental.L2SP_WEIGHT, 1.0e-4)
        self.assertFalse(hasattr(incremental, "FEATURE_SPACE_PRESERVATION_WEIGHT"))
        self.assertFalse(hasattr(incremental, "RD_WEIGHT"))
        self.assertFalse(hasattr(moe, "EXPANSION_THRESHOLD_QUANTILE"))
        self.assertFalse(hasattr(moe, "INCREMENTAL_REDUCTION"))
        self.assertFalse(hasattr(incremental, "LOCAL_RESIDUAL_PRESERVATION_WEIGHT"))
        self.assertEqual(cfg.DATASET.NAME, "CUB200")
        self.assertEqual(cfg.TRAINER.BiMC.END_SESSION, -1)


if __name__ == "__main__":
    unittest.main()
