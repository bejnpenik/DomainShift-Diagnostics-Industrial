from .early_stopping import EarlyStopper
from .config import TrainerConfig, TrainResult
from .trainer import Trainer
from .losses import coral_loss, mmd_loss, dann_loss, irm_penalty
# DomainAdaptiveTrainer, DomainGeneralizationTrainer, AdaptationConfig are imported
# directly from their modules to avoid triggering cross-package imports at package
# load time (src/ on sys.path makes ..model relative imports fail from tests).