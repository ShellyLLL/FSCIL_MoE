"""Regression tests for the complete YACS configuration schema."""

import unittest

from main import setup_cfg


class ConfigSchemaTests(unittest.TestCase):
    def test_cub200_bimc_config_merges_without_unknown_keys(self):
        # This exercises the exact two-stage merge used by main.py before any
        # model/data initialization. Unknown YAML keys must fail in CI, not on
        # the user's training machine.
        cfg = setup_cfg(
            "./configs/datasets/cub200.yaml",
            "./configs/trainers/bimc.yaml",
        )
        moe = cfg.TRAINER.BiMC.VISUAL_MOE
        incremental = cfg.TRAINER.BiMC.INCREMENTAL

        self.assertEqual(moe.DESCRIPTOR_INIT_MODE, "anchored_residual")
        self.assertAlmostEqual(moe.FEATURE_SPACE_MIN_COSINE, 0.98)
        self.assertAlmostEqual(moe.FEATURE_SPACE_MAX_ACCURACY_DROP, 0.0)
        self.assertEqual(moe.FEATURE_SPACE_ANCHORS_PER_CLASS, 2)
        self.assertEqual(moe.FEATURE_SPACE_ANCHOR_BATCH_SIZE, 16)
        self.assertAlmostEqual(incremental.FEATURE_SPACE_PRESERVATION_WEIGHT, 1.0)
        self.assertAlmostEqual(incremental.LOCAL_RESIDUAL_PRESERVATION_WEIGHT, 1.0)
        self.assertEqual(cfg.DATASET.NAME, "CUB200")
        self.assertEqual(cfg.TRAINER.BiMC.END_SESSION, -1)


if __name__ == "__main__":
    unittest.main()
