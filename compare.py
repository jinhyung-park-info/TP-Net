import argparse
import random
from common.Constants import RANDOM_SEED
from simulation_test_utils import compare_baseline_ours_simulation, load_pred_model
from real_world_test_utils import compare_baseline_ours_real_in_rendered
from common.Utils import create_directory
import os

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser(description='My Model VS Baseline')
parser.add_argument('--model_1_ver', required=False, default=13)
parser.add_argument('--model_1_type', required=False, default='global_pointnet', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--model_2_ver', required=False, default=44)
parser.add_argument('--model_2_type', required=False, default='lstm', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--model_3_ver', required=False, default=45)
parser.add_argument('--model_3_type', required=False, default='lstm', choices=['lstm', 'global_pointnet', 'local_pointnet'])

parser.add_argument('--data_1_type', required=False, default='unordered', choices=['unordered', 'ordered', 'sorted'])
parser.add_argument('--data_2_type', required=False, default='ordered', choices=['unordered', 'ordered', 'sorted'])
parser.add_argument('--data_3_type', required=False, default='sorted', choices=['unordered', 'ordered', 'sorted'])

parser.add_argument('--fine_1_ver', required=False, default=3)
parser.add_argument('--fine_2_ver', required=False, default=1)
parser.add_argument('--fine_3_ver', required=False, default=1)

parser.add_argument('--offset', required=False, default=2)
parser.add_argument('--fps', required=False, default=30)
parser.add_argument('--env', required=False, default='simulation', choices=['real', 'simulation'])
FLAGS = parser.parse_args()

MODEL_1_VER = int(FLAGS.model_1_ver)
MODEL_2_VER = int(FLAGS.model_2_ver)
MODEL_3_VER = int(FLAGS.model_3_ver)

MODEL_1_TYPE = FLAGS.model_1_type
MODEL_2_TYPE = FLAGS.model_2_type
MODEL_3_TYPE = FLAGS.model_3_type

OFFSET = int(FLAGS.offset)
FPS = int(FLAGS.fps)

DATA_TYPE_1 = FLAGS.data_1_type
DATA_TYPE_2 = FLAGS.data_2_type
DATA_TYPE_3 = FLAGS.data_3_type

FINE_VER_1 = FLAGS.fine_1_ver
FINE_VER_2 = FLAGS.fine_2_ver
FINE_VER_3 = FLAGS.fine_3_ver

ENVIRONMENT = FLAGS.env

result_savepath = create_directory(os.path.join('result', 'compare', f'{MODEL_1_TYPE}_{MODEL_1_VER}_{MODEL_2_TYPE}_{MODEL_2_VER}', f'{ENVIRONMENT}'))

if ENVIRONMENT == 'simulation':
    model_1_path = os.path.join('result', f'{MODEL_1_TYPE}', f'version_{MODEL_1_VER}', f'{MODEL_1_TYPE}_model.h5')
    model_1 = load_pred_model(model_1_path)
    model_2_path = os.path.join('result', f'{MODEL_2_TYPE}', f'version_{MODEL_2_VER}', f'{MODEL_2_TYPE}_model.h5')
    model_2 = load_pred_model(model_2_path)
    model_3_path = os.path.join('result', f'{MODEL_3_TYPE}', f'version_{MODEL_3_VER}', f'{MODEL_3_TYPE}_model.h5')
    model_3 = load_pred_model(model_3_path)
else:
    model_1 = load_pred_model(os.path.join('result', f'{MODEL_1_TYPE}', f'version_{MODEL_1_VER}', 'fine_tuning', f'version_{FINE_VER_1}', f'{MODEL_1_TYPE}_model_real_final.h5'))
    model_2 = load_pred_model(os.path.join('result', f'{MODEL_2_TYPE}', f'version_{MODEL_2_VER}', 'fine_tuning', f'version_{FINE_VER_2}', f'{MODEL_2_TYPE}_model_real_final.h5'))
    model_3 = load_pred_model(os.path.join('result', f'{MODEL_3_TYPE}', f'version_{MODEL_3_VER}', 'fine_tuning', f'version_{FINE_VER_3}', f'{MODEL_3_TYPE}_model_real_final.h5'))

random.seed(RANDOM_SEED)

if ENVIRONMENT == 'simulation':
    compare_baseline_ours_simulation(model_1, model_2, model_3, DATA_TYPE_1, DATA_TYPE_2, DATA_TYPE_3, result_savepath, OFFSET, FPS, output_video=False)
else:
    compare_baseline_ours_real_in_rendered(model_1, model_2, model_3, 3, result_savepath, OFFSET, FPS, DATA_TYPE_1, DATA_TYPE_2, DATA_TYPE_3, output_video=True)
