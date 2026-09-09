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
    cfg.TRAINER.BiMC.VISUAL_MOE.EXPANSION_MODE = 'auto'
    cfg.TRAINER.BiMC.VISUAL_MOE.INSERT_BLOCKS = [-4, -3, -2, -1]
    cfg.TRAINER.BiMC.VISUAL_MOE.NUM_BASE_EXPERTS = 4
    cfg.TRAINER.BiMC.VISUAL_MOE.BASE_REDUCTION = 4
    cfg.TRAINER.BiMC.VISUAL_MOE.BASE_SCALE_INIT = 0.5
    cfg.TRAINER.BiMC.VISUAL_MOE.INCREMENTAL_REDUCTION = 8
    cfg.TRAINER.BiMC.VISUAL_MOE.DESCRIPTOR_DIM = 64
    cfg.TRAINER.BiMC.VISUAL_MOE.DESCRIPTOR_EPS = 1.0e-4
    cfg.TRAINER.BiMC.VISUAL_MOE.TRAIN_VIEW_IDS = [0, 1, 2]
    cfg.TRAINER.BiMC.VISUAL_MOE.CALIBRATION_VIEW_IDS = [100, 101]
    cfg.TRAINER.BiMC.VISUAL_MOE.EXPANSION_Z_THRESHOLD = 1.0
    cfg.TRAINER.BiMC.VISUAL_MOE.ROUTE_BASE_Z_THRESHOLD = 1.0
    cfg.TRAINER.BiMC.VISUAL_MOE.ROUTE_REUSE_Z_THRESHOLD = 1.0
    cfg.TRAINER.BiMC.VISUAL_MOE.ROUTE_SCORE_MARGIN = 0.05
    cfg.TRAINER.BiMC.VISUAL_MOE.INCREMENTAL_SCALE_INIT = 0.5
    # The descriptor-ready check, not a negative bias, prevents premature use.
    cfg.TRAINER.BiMC.VISUAL_MOE.INCREMENTAL_GATE_BIAS_INIT = 0.5
    cfg.TRAINER.BiMC.VISUAL_MOE.MAX_NEW_EXPERTS_PER_SESSION = 1

    cfg.TRAINER.BiMC.INCREMENTAL = CN()
    cfg.TRAINER.BiMC.INCREMENTAL.EPOCHS = 6
    cfg.TRAINER.BiMC.INCREMENTAL.LOGIT_SCALE = 25.0
    cfg.TRAINER.BiMC.INCREMENTAL.LOO_ACC_THRESHOLD = 0.70
    cfg.TRAINER.BiMC.INCREMENTAL.LOO_MARGIN_THRESHOLD = 0.0
    cfg.TRAINER.BiMC.INCREMENTAL.LOO_HARD_MARGIN = 0.0
    cfg.TRAINER.BiMC.INCREMENTAL.MIN_NEW_MARGIN_GAIN = 0.0
    cfg.TRAINER.BiMC.INCREMENTAL.MARGIN_OBJECTIVE_WEIGHT = 0.05
    cfg.TRAINER.BiMC.INCREMENTAL.MIN_HISTORY_ROUTE_PRESERVATION = 1.0
    cfg.TRAINER.BiMC.INCREMENTAL.MIN_HISTORY_ACCURACY_DELTA = 0.0
    cfg.TRAINER.BiMC.INCREMENTAL.MIN_HISTORY_MARGIN_DELTA = -1.0e-3
    cfg.TRAINER.BiMC.INCREMENTAL.HISTORY_EVAL_SAMPLES_PER_CLASS = 20

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
    cfg.TRAINER.BiMC.LOSS.GATE_WEIGHT = 0.1
    cfg.TRAINER.BiMC.LOSS.HISTORY_INVARIANCE_WEIGHT = 0.1

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

def validate_cfg(cfg):
    moe = cfg.TRAINER.BiMC.VISUAL_MOE
    if not moe.ENABLE:
        return
    if list(moe.INSERT_BLOCKS) != [-4, -3, -2, -1]:
        raise ValueError("VISUAL_MOE.INSERT_BLOCKS is fixed to [-4, -3, -2, -1].")
    if int(moe.NUM_BASE_EXPERTS) != 4:
        raise ValueError("VISUAL_MOE.NUM_BASE_EXPERTS is fixed to 4.")
    if int(moe.MAX_NEW_EXPERTS_PER_SESSION) != 1:
        raise ValueError("At most one incremental expert may be proposed per session.")
    if str(moe.EXPANSION_MODE).lower() not in {"auto", "none"}:
        raise ValueError("VISUAL_MOE.EXPANSION_MODE must be 'auto' or 'none'.")
    train_views = {int(value) for value in moe.TRAIN_VIEW_IDS}
    calibration_views = {int(value) for value in moe.CALIBRATION_VIEW_IDS}
    if not train_views or not calibration_views or train_views & calibration_views:
        raise ValueError("Descriptor train/calibration view ids must be non-empty and disjoint.")
    if not str(cfg.MODEL.BACKBONE.NAME).startswith("ViT-"):
        raise ValueError("Visual MoE insertion currently requires a CLIP ViT backbone.")
    if not cfg.TRAINER.BiMC.FREEZE_TEXT_ENCODER or not cfg.TRAINER.BiMC.FREEZE_VISUAL_BACKBONE:
        raise ValueError("Continual routing requires both pretrained encoders to remain frozen.")


def setup_cfg(dataset_cfg_file, method_cfg_file, opts=None):
    cfg = CN()
    extend_cfg(cfg)
    # YACS opens paths with the platform default encoding.  Trainer files carry
    # Chinese comments and are UTF-8, so explicitly load them to keep Windows
    # runs reproducible instead of depending on the active code page.
    for cfg_file in (dataset_cfg_file, method_cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as file:
            cfg.merge_from_other_cfg(CN.load_cfg(file))
    if opts:
        cfg.merge_from_list(opts)
    validate_cfg(cfg)
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument('--data_cfg', type=str, required=True, help="Path to the data configuration file")
    parser.add_argument('--train_cfg', type=str, required=True, help="Path to the training configuration file")
    parser.add_argument('--resume', type=str, default='', help='Session checkpoint to resume from')
    parser.add_argument('--support_manifest', type=str, default='', help='Optional fixed support manifest JSON')
    parser.add_argument('--end_session', type=int, default=-1, help='Stop after this FSCIL session')
    parser.add_argument('--expansion_mode', choices=('auto', 'none'), default=None,
                        help='Enable demand-driven expert proposals or disable expansion')
    parser.add_argument('opts', nargs=argparse.REMAINDER, help='Additional KEY VALUE YACS overrides')
    args = parser.parse_args()
    overrides = list(args.opts)
    if args.resume:
        overrides += ['TRAINER.BiMC.CHECKPOINT.RESUME', args.resume]
    if args.support_manifest:
        overrides += ['DATASET.SUPPORT_MANIFEST_PATH', args.support_manifest]
    if args.end_session >= 0:
        overrides += ['TRAINER.BiMC.END_SESSION', str(args.end_session)]
    if args.expansion_mode is not None:
        overrides += ['TRAINER.BiMC.VISUAL_MOE.EXPANSION_MODE', args.expansion_mode]
    cfg = setup_cfg(args.data_cfg, args.train_cfg, overrides)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    
    from engine.engine import Runner
    engine = Runner(cfg)
    engine.run()

if __name__ == '__main__':
    main()
