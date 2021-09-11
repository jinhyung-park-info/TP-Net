import argparse
import os
import tensorflow as tf
import numpy as np
import random
from common.Utils import create_directory


def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--num_input', required=False, default=4)
    parser.add_argument('--epoch', required=False, default=100)
    parser.add_argument('--batch_size', required=False, default=128)
    parser.add_argument('--lr', required=False, default=0.001)

    parser.add_argument('--train_verbose', required=False, default=2)
    parser.add_argument('--cp_verbose', required=False, default=1)

    parser.add_argument('--beta_1', required=False, default=0.99)
    parser.add_argument('--beta_2', required=False, default=0.999)
    parser.add_argument('--epsilon', required=False, default=1e-07)
    parser.add_argument('--use_amsgrad', required=False, default=0)

    parser.add_argument('--num_output', required=False, default=8)
    parser.add_argument('--use_transform_net', required=False, default=1)
    parser.add_argument('--n_global_features', required=False, default=128)
    parser.add_argument('--bn', required=False, default=0)

    parser.add_argument('--data_offset', required=False, default=4)
    parser.add_argument('--data_type', required=False, default='ordered', choices=['ordered', 'unordered', 'sorted'])

    parser.add_argument('--n_dims', required=False, default=2)
    parser.add_argument('--n_particles', required=False, default=30)

    parser.add_argument('--save_path', required=False, default='./result')
    parser.add_argument('--data_path', required=False, default='./data/simulation')
    parser.add_argument('--n_train_data', required=False, default=-1)
    parser.add_argument('--n_val_data', required=False, default=-1)

    args = parser.parse_args()

    args.save_path = create_directory(os.path.join(args.save_path, 'tp_net', f'tp_net-{args.num_input}', f'seed_{args.seed}'))
    args.data_path = os.path.join(args.data_path, f'input_{args.num_input}', f'offset_{args.data_offset}_num_pred_{args.num_output}')
    return args


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
