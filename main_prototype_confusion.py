"""Entry point for Prototype-Confusion Guided Dynamic MoE-FSCIL."""
import argparse
from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
from main import extend_cfg

def extend_pcmoe_cfg(cfg):
    cfg.TRAINER.PCMOE = CN()
    p = cfg.TRAINER.PCMOE
    p.END_SESSION = -1
    p.TEXT_TEMPLATES = []
    p.VISUAL_MOE = CN()
    p.VISUAL_MOE.INSERT_BLOCKS = [-4, -3, -2, -1]
    p.VISUAL_MOE.NUM_BASE_EXPERTS = 4
    p.VISUAL_MOE.BASE_REDUCTION = 4
    p.VISUAL_MOE.INCREMENTAL_REDUCTION = 16
    p.VISUAL_MOE.BASE_SCALE_INIT = 0.5
    p.VISUAL_MOE.TOP_K = 2
    p.CALIBRATION = CN()
    p.CALIBRATION.INITIAL_TEMPERATURE_VISUAL = 0.1
    p.CALIBRATION.INITIAL_TEMPERATURE_TEXT = 0.1
    p.CALIBRATION.MIN_TEMPERATURE = 0.02
    p.CALIBRATION.MAX_TEMPERATURE = 2.0
    p.CALIBRATION.RELIABILITY_KAPPA = 2.0
    p.CALIBRATION.LAMBDA_VISUAL = 0.1
    p.CALIBRATION.VISUAL_TAU = 16.0
    p.CALIBRATION.LAMBDA_TEXT = 0.5
    p.CALIBRATION.INTER_MODAL_BETA = 0.3
    p.EXPANSION = CN()
    p.EXPANSION.APC_MIN_PERCENTILE = 0.8
    p.EXPANSION.MAX_LAYERS_PER_SESSION = 4
    p.EXPANSION.MIN_RISK_GAIN = 0.001
    p.EXPANSION.MAX_OLD_CLASS_RISK_INCREASE = 0.01
    p.OPTIM = CN()
    p.OPTIM.BASE_EPOCHS = 50
    p.OPTIM.BASE_BATCH_SIZE = 64
    p.OPTIM.BASE_LR = 1e-4
    p.OPTIM.INCREMENTAL_EPOCHS = 10
    p.OPTIM.INCREMENTAL_LR = 3e-4
    p.OPTIM.WEIGHT_DECAY = 1e-4
    p.LOSS = CN()
    p.LOSS.ROUTE_BALANCE_WEIGHT = 0.01
    p.CHECKPOINT = CN()
    p.CHECKPOINT.DIR = 'checkpoints/prototype_confusion_moe'
    p.CHECKPOINT.RESUME = ''
    return cfg

def main():
    parser = argparse.ArgumentParser(description='Prototype-Confusion Guided Dynamic MoE-FSCIL')
    parser.add_argument('--data_cfg', required=True)
    parser.add_argument('--train_cfg', required=True)
    parser.add_argument('--resume', default='')
    parser.add_argument('--end_session', type=int, default=-1)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = extend_pcmoe_cfg(extend_cfg(CN()))
    for path in (args.data_cfg, args.train_cfg):
        with open(path, 'r', encoding='utf-8') as handle:
            cfg.merge_from_other_cfg(CN.load_cfg(handle))
    overrides = list(args.opts)
    if args.resume:
        overrides += ['TRAINER.PCMOE.CHECKPOINT.RESUME', args.resume]
    if args.end_session >= 0:
        overrides += ['TRAINER.PCMOE.END_SESSION', str(args.end_session)]
    if overrides:
        cfg.merge_from_list(overrides)
    cfg.freeze()
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    from engine.prototype_confusion_engine import PrototypeConfusionRunner
    PrototypeConfusionRunner(cfg).run()

if __name__ == '__main__':
    main()
