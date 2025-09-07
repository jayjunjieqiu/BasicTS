import os
import sys
from argparse import ArgumentParser

# Ensure project root is on path and as cwd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import basicts


def parse_args():
    parser = ArgumentParser(description='Train HF-PPE and optionally evaluate with a different horizon.')
    parser.add_argument('--dataset', type=str, default='Electricity', choices=['Electricity', 'ETTh1'], help='Dataset name')
    parser.add_argument('--gpus', type=str, default='0', help='Visible GPU ids, e.g., "0" or "0,1"')
    parser.add_argument('--train_horizon', type=int, default=24, help='Training horizon H')
    parser.add_argument('--test_horizon', type=int, default=0, help='Optional evaluation horizon H* (0 to skip)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--m', type=int, default=4, help='Number of harmonics')
    parser.add_argument('--ctx_len', type=int, default=24, help='Context length for hyper-network')
    parser.add_argument('--lambda_unit', type=float, default=1e-3, help='Unit-circle reg weight')
    parser.add_argument('--lambda_phase', type=float, default=1e-3, help='Phase-smoothness reg weight')
    parser.add_argument('--lambda_cycle', type=float, default=1e-3, help='Cycle-consistency reg weight')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber delta for phase-smoothness')
    return parser.parse_args()


def main():
    args = parse_args()

    # choose baseline config
    cfg_path = f"baselines/HFPPE/{args.dataset}.py"

    # inject training env overrides
    os.environ['HFPPE_OUTPUT_LEN'] = str(args.train_horizon)
    os.environ['HFPPE_EPOCHS'] = str(args.epochs)
    os.environ['HFPPE_LAMBDA_UNIT'] = str(args.lambda_unit)
    os.environ['HFPPE_LAMBDA_PHASE'] = str(args.lambda_phase)
    os.environ['HFPPE_LAMBDA_CYCLE'] = str(args.lambda_cycle)
    os.environ['HFPPE_HUBER_DELTA'] = str(args.huber_delta)

    # Train
    basicts.launch_training(cfg=cfg_path, gpus=args.gpus)

    # Optional evaluation with different horizon H*
    if args.test_horizon and args.test_horizon > 0:
        os.environ['HFPPE_OUTPUT_LEN'] = str(args.test_horizon)
        basicts.launch_evaluation(cfg=cfg_path, ckpt_path='', device_type='gpu', gpus=args.gpus, batch_size=None)


if __name__ == '__main__':
    main()

