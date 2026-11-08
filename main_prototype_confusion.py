"""Entry point for Prototype-Confusion Guided Dynamic MoE-FSCIL."""
from yacs.config import CfgNode as CN
import argparse

from utils.util import set_gpu, set_seed


def main():
    parser = argparse.ArgumentParser(description="Prototype-confusion guided dynamic MoE FSCIL")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    args = parser.parse_args()
    cfg = CN()
    for path in (args.data_cfg, args.train_cfg):
        with open(path, "r", encoding="utf-8") as handle:
            cfg.merge_from_other_cfg(CN.load_cfg(handle))
    cfg.freeze()
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    from engine.prototype_confusion_engine import PrototypeConfusionRunner
    PrototypeConfusionRunner(cfg).run()


if __name__ == "__main__":
    main()
