import argparse
import os
import random
from common.Constants import RANDOM_SEED
from simulation_test_utils import generate_simulation_result_videos, load_pred_model
from real_world_test_utils import generate_real_result_videos, generate_fine_tuning_result_videos, generate_rendered_videos
from common.Utils import create_directory

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser(description='Model Testing Options')
parser.add_argument('--model_ver', required=False, default=25)
parser.add_argument('--real_model_ver', required=False, default=2)
parser.add_argument('--model_type', required=False, default='lstm', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--num_input', required=False, default=3)
parser.add_argument('--num_output', required=False, default=8)
parser.add_argument('--offset', required=False, default=2)
parser.add_argument('--length', required=False, default=100)
parser.add_argument('--fps', required=False, default=30)
parser.add_argument('--test_type', required=False, default='test', choices=['test', 'trained'])
parser.add_argument('--test_data_type', required=False, default='ordered', choices=['ordered', 'unordered', 'sorted'])
parser.add_argument('--env', required=False, default='rendered', choices=['simulation', 'real', 'both', 'rendered', 'simulation_all'])
FLAGS = parser.parse_args()

SIM_MODEL_VER = int(FLAGS.model_ver)
REAL_MODEL_VER = int(FLAGS.real_model_ver)
MODEL_TYPE = FLAGS.model_type
NUM_INPUT = int(FLAGS.num_input)
NUM_OUTPUT = int(FLAGS.num_output)
OFFSET = int(FLAGS.offset)
FPS = int(FLAGS.fps)
LENGTH = int(FLAGS.length)
ENVIRONMENT = FLAGS.env
TEST_DATA_TYPE = FLAGS.test_data_type
TEST_TYPE = FLAGS.test_type

result_base_path = os.path.join('result', MODEL_TYPE, f'version_{SIM_MODEL_VER}')
prediction_model_path = os.path.join(result_base_path, f'{MODEL_TYPE}_model.h5')
prediction_model = load_pred_model(prediction_model_path)
random.seed(RANDOM_SEED)

if ENVIRONMENT == 'real':
    video_savepath = create_directory(os.path.join(result_base_path, 'real_world_prediction_result', f'offset_{OFFSET}'))
    generate_real_result_videos(prediction_model, NUM_INPUT, video_savepath, OFFSET, FPS, TEST_DATA_TYPE)
elif ENVIRONMENT == 'simulation':
    video_savepath = create_directory(os.path.join(result_base_path, 'simulation_prediction_result', f'offset_{OFFSET}_{TEST_TYPE}_cases'))
    generate_simulation_result_videos(prediction_model, video_savepath, OFFSET, LENGTH, FPS, TEST_TYPE, TEST_DATA_TYPE)
elif ENVIRONMENT == 'rendered':
    video_savepath = create_directory(os.path.join(result_base_path, 'fine_tuning_result_rendered', f'offset_{OFFSET}_version_{REAL_MODEL_VER}'))
    simulation_model = prediction_model
    real_model = load_pred_model(os.path.join(result_base_path, 'fine_tuning', f'version_{REAL_MODEL_VER}', f'{MODEL_TYPE}_model_real_final.h5'))
    generate_rendered_videos(simulation_model, real_model, NUM_INPUT, video_savepath, OFFSET, FPS, TEST_DATA_TYPE)
else:
    video_savepath = create_directory(os.path.join(result_base_path, 'fine_tuning_result', f'offset_{OFFSET}_version_{REAL_MODEL_VER}'))
    simulation_model = prediction_model
    real_model = load_pred_model(os.path.join(result_base_path, 'fine_tuning', f'version_{REAL_MODEL_VER}', f'{MODEL_TYPE}_model_real_final.h5'))
    generate_fine_tuning_result_videos(simulation_model, real_model, NUM_INPUT, video_savepath, OFFSET, FPS, TEST_DATA_TYPE)
