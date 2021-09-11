from tensorflow.keras.models import load_model
import numpy as np
from common.Constants import *
from common.Utils import load_json, normalize_pointset, get_nested_pointset, normalize_nested_pointset, denormalize_dnri_pointset, sort_pointset_by_ascending_x, sort_pointset_by_descending_y, center_transform
from tqdm import tqdm
from loss import get_cd_loss
from random import sample


def preprocess_dnri_predictions(predictions):
    predictions = predictions[:, :, :, :, :2]             # discard predicted velocity
    for test_case in range(60):                           # for 60 simulation test cases
        for timestep in range(len(predictions[0][0])):    # change the metrics from -1~1 to 0~1
            predictions[test_case][0][timestep] = np.array(normalize_nested_pointset(denormalize_dnri_pointset(predictions[test_case][0][timestep])))

    return predictions.reshape(60, -1, 30, 2)


def get_error_for_sim_data(model_type, seed, num_input, test_length, error_type, data_type='ordered'):
    if model_type == 'static_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_static_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dynamic_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_dynamic_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dnri':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dynamics_prior':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')
        predictions = predictions[:, :, :, :2].tolist()

    elif model_type == 'graphrnn':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions-num_samples_15_{data_type}.npy')
        predictions = predictions[:, :, :, :2]

    else:  # Ours (TP-Net)
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')

    errors = []
    dnri_offset = 1
    if model_type == 'dnri':
        dnri_offset = 0

    if error_type == 'Position':
        for i, test_case in tqdm(list(enumerate(SIM_DATA_EVAL_CASES))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * SIM_DATA_OFFSET]], dtype='float32')
                errors[i].append(get_cd_loss(ground_truth, np.array([predictions[i][dnri_offset + timestep]], dtype='float32')))

    else:  # error_type == 'Shape'
        for i, test_case in tqdm(list(enumerate(SIM_DATA_EVAL_CASES))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * SIM_DATA_OFFSET]], dtype='float32')
                transformed_ground_truth = center_transform(ground_truth)
                transformed_prediction = center_transform(np.array([predictions[i][dnri_offset + timestep]], dtype='float32'))
                errors[i].append(get_cd_loss(transformed_ground_truth, transformed_prediction))

    return errors


def load_pred_model(pred_model_path):
    print('================ Loading Model ================')
    return load_model(filepath=pred_model_path, compile=False)


def get_simulation_input_pointset(test_case_info, num_input, init_data_type):
    force, angle, init_x_pos, init_y_pos = test_case_info
    pointset, ptr = load_json(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()
    input_pointset = [normalize_pointset(pointset[i]) for i in range(0, SIM_DATA_OFFSET * num_input, SIM_DATA_OFFSET)]

    if init_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif init_data_type == 'sorted_x':
        input_pointset = [sort_pointset_by_ascending_x(pointset) for pointset in input_pointset]
    elif init_data_type == 'sorted_y':
        input_pointset = [sort_pointset_by_descending_y(pointset) for pointset in input_pointset]

    return np.array([input_pointset])


def get_ground_truth_pointset(test_case_info):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    return [get_nested_pointset(pointset) for pointset in pointsets]
