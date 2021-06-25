import argparse
import os
import random
from common.Constants import RANDOM_SEED, DNRI_TEST_CASES, NUM_PARTICLES
from utils import *
from common.Utils import create_directory
from tqdm import tqdm


def get_simulation_input_pointset_for_dnri_comparison(test_case_info, offset, test_data_type, num_input, start_timestep):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointset, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    # Prepare initial input for predicting this animation
    input_pointset = [normalize_pointset(pointset[i]) for i in range(offset * start_timestep, offset * (start_timestep + num_input), offset)]
    if test_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif test_data_type == 'sorted':
        input_pointset = [sort_pointset(pointset) for pointset in input_pointset]

    return np.array([input_pointset])  # shape == (1, NUM_INPUT_FRAMES, 30, 2)


# ========================= Model Constants ====================================

parser = argparse.ArgumentParser(description='Model Testing Options')
parser.add_argument('--model_ver', required=False, default=45)
parser.add_argument('--real_model_ver', required=False, default=0)
parser.add_argument('--model_type', required=False, default='lstm', choices=['lstm', 'global_pointnet'])
parser.add_argument('--num_input', required=False, default=3)
parser.add_argument('--start_timestep', required=False, default=0)
parser.add_argument('--offset', required=False, default=4)
parser.add_argument('--length', required=False, default=150)
parser.add_argument('--test_data_type', required=False, default='sorted', choices=['ordered', 'unordered', 'sorted'])
parser.add_argument('--env', required=False, default='simulation', choices=['simulation', 'real'])
FLAGS = parser.parse_args()

SIM_MODEL_VER = int(FLAGS.model_ver)
REAL_MODEL_VER = int(FLAGS.real_model_ver)
MODEL_TYPE = FLAGS.model_type
NUM_INPUT = int(FLAGS.num_input)
START_TIMESTEP = int(FLAGS.start_timestep)
OFFSET = int(FLAGS.offset)
LENGTH = int(FLAGS.length)
ENVIRONMENT = FLAGS.env
TEST_DATA_TYPE = FLAGS.test_data_type

result_base_path = os.path.join('../result', MODEL_TYPE, f'version_{SIM_MODEL_VER}')
prediction_model_path = os.path.join(result_base_path, f'{MODEL_TYPE}_model.h5')
prediction_model = load_pred_model(prediction_model_path)
random.seed(RANDOM_SEED)

npy_savepath = create_directory(os.path.join(result_base_path, 'simulation_prediction_result', f'offset_{OFFSET}_test_cases'))
all_predictions = []

for i, test_case in enumerate(DNRI_TEST_CASES[:100]):
    print(f'============== Testing Case #{i + 1} ==============')
    predictions = []
    input_info = get_simulation_input_pointset_for_dnri_comparison(test_case, OFFSET, TEST_DATA_TYPE, NUM_INPUT, START_TIMESTEP)

    for _ in tqdm(range(LENGTH)):
        predicted_pointset = prediction_model.predict(input_info)[0]
        assert predicted_pointset.shape == (1, NUM_PARTICLES, 2), 'Expected {} but received {}'.format((1, NUM_PARTICLES, 2), predicted_pointset.shape)
        predictions.append(predicted_pointset[0])
        input_info = update_input_pointset(input_info, predicted_pointset)

    all_predictions.append(predictions)

np.save(os.path.join(npy_savepath, f'softbody_predictions_ours_start_{START_TIMESTEP}.npy'), np.array(all_predictions))
