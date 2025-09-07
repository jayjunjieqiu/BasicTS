import os
import sys
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import HFPPE, hf_ppe_loss

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'Electricity'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
# default training horizon 24; allow env override for convenience
OUTPUT_LEN = int(os.environ.get('HFPPE_OUTPUT_LEN', 24))
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture and parameters
MODEL_ARCH = HFPPE
NUM_NODES = 321
MODEL_PARAM = {
    'enc_in': NUM_NODES,
    'seq_len': INPUT_LEN,
    'pred_len': OUTPUT_LEN,  # for API consistency only; inference supports arbitrary H*
    'd_model': 128,
    'n_heads': 4,
    'e_layers': 2,
    'd_ff': 256,
    'dropout': 0.1,
    'm': 4,          # number of harmonics
    'ctx_len': 24,   # context length for hyper-network
    'use_y_in_encoder': False,
}

# Regularizer weights for loss composition
LAMBDA_UNIT = float(os.environ.get('HFPPE_LAMBDA_UNIT', 1e-3))
LAMBDA_PHASE = float(os.environ.get('HFPPE_LAMBDA_PHASE', 1e-3))
LAMBDA_CYCLE = float(os.environ.get('HFPPE_LAMBDA_CYCLE', 1e-3))
HUBER_DELTA = float(os.environ.get('HFPPE_HUBER_DELTA', 1.0))

NUM_EPOCHS = int(os.environ.get('HFPPE_EPOCHS', 100))

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'HF-PPE on Electricity'
CFG.GPU_NUM = 1
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
# Forward features: just use target; PPE already encodes timing
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MSE': masked_mse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = hf_ppe_loss
CFG.TRAIN.LOSS_ARGS = {
    'lambda_unit': LAMBDA_UNIT,
    'lambda_phase': LAMBDA_PHASE,
    'lambda_cycle': LAMBDA_CYCLE,
    'huber_delta': HUBER_DELTA,
}
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'Adam'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 2e-4,
    'weight_decay': 1e-4,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'milestones': [1, 25, 50],
    'gamma': 0.5,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.USE_GPU = True

