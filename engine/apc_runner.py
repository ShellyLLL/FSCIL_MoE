"""Entry point for the unified FSCIL dynamic-MoE runner."""
from engine.engine import Runner


class APCRunner(Runner):
    """Compatibility entry point; expansion and training share one unified pipeline."""
    pass
