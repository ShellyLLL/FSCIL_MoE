from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse

def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")

def extend_cfg(cfg):
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ''
    cfg.DEVICE.GPU_ID = ''
    cfg.METHOD = ''
    cfg.SEED = -1

    cfg.DATASET = CN()
    cfg.DATASET.NAME = ''
    cfg.DATASET.ROOT = ''
    cfg.DATASET.GPT_PATH = ''
    cfg.DATASET.NUM_CLASSES   = -1
    cfg.DATASET.NUM_INIT_CLS  = -1
    cfg.DATASET.NUM_INC_CLS   = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT  = -1
    cfg.DATASET.BETA = -1.0
    # Session support examples are sampled once by DatasetManager. An empty
    # path auto-saves a dataset/seed-specific manifest; set an explicit path
    # to share it between variants. -1 derives the private manifest RNG seed
    # from SEED.
    cfg.DATASET.SUPPORT_MANIFEST_PATH = ''
    cfg.DATASET.SUPPORT_MANIFEST_SEED = -1
    cfg.DATASET.SUPPORT_MANIFEST_AUTO_SAVE = True
    
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ''

    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ''
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0

    cfg.TRAINER.BiMC.FREEZE_TEXT_ENCODER = True
    cfg.TRAINER.BiMC.FREEZE_VISUAL_BACKBONE = True

    cfg.TRAINER.BiMC.VISUAL_MOE = CN()
    cfg.TRAINER.BiMC.VISUAL_MOE.ENABLE = False
    cfg.TRAINER.BiMC.VISUAL_MOE.INSERT_BLOCKS = [-4, -3, -2, -1]
    cfg.TRAINER.BiMC.VISUAL_MOE.NUM_BASE_EXPERTS = 4
    cfg.TRAINER.BiMC.VISUAL_MOE.BASE_REDUCTION = 4
    cfg.TRAINER.BiMC.VISUAL_MOE.BASE_SCALE_INIT = 0.5
    cfg.TRAINER.BiMC.VISUAL_MOE.INCREMENTAL_REDUCTION = 48
    cfg.TRAINER.BiMC.VISUAL_MOE.DESCRIPTOR_DIM = 64
    cfg.TRAINER.BiMC.VISUAL_MOE.DESCRIPTOR_EPS = 1.0e-4

    cfg.TRAINER.BiMC.INCREMENTAL = CN()
    cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS = 6
    cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE = 100.0
    cfg.TRAINER.BiMC.INCREMENTAL.STABILITY_WEIGHT = 0.0

    cfg.TRAINER.BiMC.CHECKPOINT = CN()
    cfg.TRAINER.BiMC.CHECKPOINT.ENABLE = True
    cfg.TRAINER.BiMC.CHECKPOINT.DIR = 'checkpoints/fscil_moe'
    cfg.TRAINER.BiMC.CHECKPOINT.RESUME = ''
    cfg.TRAINER.BiMC.CHECKPOINT.SAVE_EVERY_SESSION = True

    cfg.TRAINER.BiMC.OPTIM = CN()
    cfg.TRAINER.BiMC.OPTIM.BASE_EPOCHS = 50
    cfg.TRAINER.BiMC.OPTIM.BASE_DESCRIPTOR_EPOCHS = 8
    cfg.TRAINER.BiMC.OPTIM.INCREMENTAL_DESCRIPTOR_EPOCHS = 5
    cfg.TRAINER.BiMC.OPTIM.LR_BASE_MOE = 1e-4
    cfg.TRAINER.BiMC.OPTIM.LR_INCREMENTAL_EXPERT = 3e-4
    cfg.TRAINER.BiMC.OPTIM.LR_INCREMENTAL_ROUTER = 3e-4
    cfg.TRAINER.BiMC.OPTIM.LR_DESCRIPTOR = 1e-3
    cfg.TRAINER.BiMC.OPTIM.WEIGHT_DECAY = 1e-4
    cfg.TRAINER.BiMC.OPTIM.BETAS = (0.9, 0.999)

    cfg.TRAINER.BiMC.LOSS = CN()
    cfg.TRAINER.BiMC.LOSS.CE_WEIGHT = 1.0
    cfg.TRAINER.BiMC.LOSS.LB_WEIGHT = 0.01
    cfg.TRAINER.BiMC.LOSS.KD_WEIGHT = 0.5

    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT = CN()
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.ENABLE = False
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.EPOCHS = 30
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.LR = 1.0e-2
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.OLD_PROTECT_WEIGHT = 1.0
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.ANCHOR_WEIGHT = 1.0
    cfg.TRAINER.BiMC.PROTOTYPE_REFINEMENT.OLD_PROTECT_MARGIN = 0.70
    cfg.TRAINER.BiMC.FUSED_CONFLICT_STRENGTH = 0.0
    cfg.TRAINER.BiMC.FUSED_CONFLICT_MARGIN = 0.05
    cfg.TRAINER.BiMC.END_SESSION = -1

def extend_pcmoe_cfg(cfg):
    """Register PC-MoE options in the shared entry-point configuration."""
    cfg.TRAINER.PCMOE = CN()
    p = cfg.TRAINER.PCMOE
    p.ENABLE = False
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
    p.EXPANSION.APC_MIN_PERCENTILE = 0.95
    p.EXPANSION.MAX_LAYERS_PER_SESSION = 4
    p.EXPANSION.MIN_RISK_GAIN = 0.001
    p.EXPANSION.MIN_DEMAND_RISK_GAIN = 0.001
    p.EXPANSION.MAX_OLD_CLASS_RISK_INCREASE = 0.01
    p.EXPANSION.MAX_NEW_CLASS_RISK_INCREASE = 0.01
    p.OPTIM = CN()
    p.OPTIM.BASE_EPOCHS = 50
    p.OPTIM.BASE_BATCH_SIZE = 64
    p.OPTIM.BASE_LR = 1e-4
    p.OPTIM.INCREMENTAL_EPOCHS = 10
    p.OPTIM.INCREMENTAL_LR = 3e-4
    p.OPTIM.WEIGHT_DECAY = 1e-4
    p.LOSS = CN()
    p.LOSS.ROUTE_BALANCE_WEIGHT = 0.01
    p.LOSS.ROUTE_MARGIN = 0.05
    p.LOSS.NEW_ROUTE_WEIGHT = 1.0
    p.LOSS.HISTORICAL_ROUTE_WEIGHT = 1.0
    p.LOSS.MAX_HISTORICAL_ROUTE_DRIFT = 0.05
    p.LOSS.MAX_HISTORICAL_INTRUSION_RATE = 0.01
    p.LOSS.MIN_DEMAND_ACTIVATION_RATE = 0.20
    p.LOSS.MAX_NONDEMAND_ACTIVATION_RATE = 0.01
    p.CHECKPOINT = CN()
    p.CHECKPOINT.DIR = 'checkpoints/prototype_confusion_moe'
    p.CHECKPOINT.RESUME = ''


def setup_cfg(dataset_cfg_file, method_cfg_file, opts=None):
    cfg = CN()
    extend_cfg(cfg)
    extend_pcmoe_cfg(cfg)
    # YACS opens paths with the platform default encoding.  Trainer files carry
    # Chinese comments and are UTF-8, so explicitly load them to keep Windows
    # runs reproducible instead of depending on the active code page.
    for cfg_file in (dataset_cfg_file, method_cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as file:
            cfg.merge_from_other_cfg(CN.load_cfg(file))
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument('--data_cfg', type=str, required=True, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, required=True, help="Path to the training configuration file")
    parser.add_argument('--resume', type=str, default='', help='Session checkpoint to resume from')
    parser.add_argument('--support_manifest', type=str, default='', help='Optional fixed support manifest JSON')
    parser.add_argument('--end_session', type=int, default=-1, help='Stop after this FSCIL session')
    parser.add_argument('opts', nargs=argparse.REMAINDER, help='Additional KEY VALUE YACS overrides')
    args = parser.parse_args()
    overrides = list(args.opts)
    if args.resume:
        overrides += ['TRAINER.BiMC.CHECKPOINT.RESUME', args.resume]
        overrides += ['TRAINER.PCMOE.CHECKPOINT.RESUME', args.resume]
    if args.support_manifest:
        overrides += ['DATASET.SUPPORT_MANIFEST_PATH', args.support_manifest]
    if args.end_session >= 0:
        overrides += ['TRAINER.BiMC.END_SESSION', str(args.end_session)]
        overrides += ['TRAINER.PCMOE.END_SESSION', str(args.end_session)]
    cfg = setup_cfg(args.data_cfg, args.train_cfg, overrides)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    
    if cfg.TRAINER.PCMOE.ENABLE:
        from engine.prototype_confusion_engine import PrototypeConfusionRunner
        engine = PrototypeConfusionRunner(cfg)
    else:
        from engine.engine import Runner
        engine = Runner(cfg)
    engine.run()

if __name__ == '__main__':
    main()
