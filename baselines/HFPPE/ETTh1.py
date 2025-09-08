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
DATA_NAME = 'ETTh1'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

# Model architecture and parameters
MODEL_ARCH = HFPPE
NUM_NODES = regular_settings.get('NUM_NODES', 7)  # ETTh datasets typically 7 variables
MODEL_PARAM = {
    'enc_in': NUM_NODES,
    'seq_len': INPUT_LEN,
    'pred_len': OUTPUT_LEN,
    'd_model': 128,
    'n_heads': 4,
    'e_layers': 2,
    'd_ff': 256,
    'dropout': 0.1,
    'm': 4,
    'ctx_len': 336,
    'use_y_in_encoder': False,
    'compute_phase_reg': 0,
    'compute_cycle_reg': 0,
    'reg_stride': 8,
    # new stabilization knobs
    'a_max': 1.5,
    'period_min': 12,
    'period_max': 336,
    'use_scaled_r': 0,
    'pos_dim': 8,
    'use_query_context': 1,
    # RevIN
    'revin': 1,
    'revin_affine': 0,
    'revin_subtract_last': 0,
    # Fixed history feature dimension to avoid lazy layers
    'hist_feat_dim': 1,
    # PPE LN off for stronger signal
    'apply_ppe_ln': 1,
    # Expert feedback toggles
    'hnet_type': 'mlp',
    'phi_mode': 'sincos',
    'wave_mlp_hidden': 0,
    'ppe_code_dim': 0,
    'use_query_gate': 1,
    'query_pos_dim': 4,
    'simple': 0,
}

LAMBDA_UNIT = float(os.environ.get('HFPPE_LAMBDA_UNIT', 0.0))
LAMBDA_PHASE = float(os.environ.get('HFPPE_LAMBDA_PHASE', 1e-3))
LAMBDA_CYCLE = float(os.environ.get('HFPPE_LAMBDA_CYCLE', 1e-3))
HUBER_DELTA = float(os.environ.get('HFPPE_HUBER_DELTA', 1.0))

NUM_EPOCHS = int(os.environ.get('HFPPE_EPOCHS', 100))

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'HF-PPE on ETTh1'
CFG.GPU_NUM = 4
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
    'lambda_fft': 0.0,
}
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'AdamW'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 5e-4,
    'betas': (0.9, 0.95),
    'weight_decay': 1e-2,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = 'CosineWarmup'
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': 10,
    'num_training_steps': NUM_EPOCHS,
    'num_cycles': 0.5,
}
CFG.TRAIN.CL = EasyDict({'WARM_EPOCHS': 5, 'CL_EPOCHS': 5, 'PREDICTION_LENGTH': OUTPUT_LEN, 'STEP_SIZE': 8})
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
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
