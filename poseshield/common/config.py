import logging
import os
import yaml
from yacs.config import CfgNode as CN

# Load global configuration paths
global_config_path_ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config_files",
    "global_path.yaml"
)
with open(global_config_path_, "r") as f:
    global_config_ = yaml.safe_load(f)

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Directory containing .npz files
_C.DATA.DIR = global_config_.get("DATA_PATH", "dataset")

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Number of epochs for the initial training phase
_C.TRAIN.NUM_EPOCHS = 200
# Batch size for DataLoader
_C.TRAIN.BATCH_SIZE = 128
# Learning rate for the model optimizer
_C.TRAIN.LR = 2e-4
# Number of epochs between validations
_C.TRAIN.VAL_INTERVAL = 20

# Loss weights and dt parameters for poseshield_loss
_C.TRAIN.GRAD_LOSS_WEIGHT = 0.0
_C.TRAIN.TD_LOSS_WEIGHT = 0.1
_C.TRAIN.DT = 0.01
# Experiment name (used for logging and checkpoint folders)
_C.TRAIN.EXP_NAME = "train_basic_large"

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Input feature dimension (21 joints × 6 values each)
_C.MODEL.IN_DIM = 21 * 6
# Hidden layer size of the MLP
_C.MODEL.HIDDEN_DIM = 256
# Number of layers in the MLP
_C.MODEL.NUM_LAYERS = 6
_C.MODEL.ACTIVATION = "relu"  # Model architecture type

def get_cfg_defaults():
    """
    Retrieve a fresh copy of the default configuration.
    Returns:
        CfgNode: a cloned copy of the default config
    """
    cfg = _C.clone()
    cfg.set_new_allowed(True)
    return cfg

def logging_config(log_file):
    # 1. Get or create a logger
    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)

    # 2. Create a file handler that logs messages to app.log
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)  # Only log INFO and above

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                  datefmt="%m-%d %H:%M")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger