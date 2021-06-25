from tensorflow.keras.models import load_model
from common.Utils import load_json, normalize_pointset, sort_pointset
import os
from random import sample
import numpy as np
from common.Constants import RAW_DATA_PATH


def load_pred_model(pred_model_path):
    print('================ Loading Model ================')
    return load_model(filepath=pred_model_path, compile=False)


def get_simulation_input_pointset(test_case_info, offset, test_data_type, num_input, start_timestep):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointset, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    # Prepare initial input for predicting this animation
    input_pointset = [normalize_pointset(pointset[i]) for i in range(start_timestep, start_timestep + offset * num_input, offset)]
    if test_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif test_data_type == 'sorted':
        input_pointset = [sort_pointset(pointset) for pointset in input_pointset]

    return np.array([input_pointset])  # shape == (1, NUM_INPUT_FRAMES, 30, 2)


def update_input_pointset(old_input, predicted_pointset):
    return np.expand_dims(np.concatenate([old_input[0][1:], predicted_pointset], axis=0), axis=0) # shape == (1, NUM_INPUT_FRAMES, 20, 2)
