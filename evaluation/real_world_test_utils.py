import numpy as np
from common.Constants import *
from common.Utils import load_json, sort_pointset_by_ascending_x, sort_pointset_by_descending_y, center_transform
from tqdm import tqdm
import re
from random import sample
from loss import get_cd_loss
import cv2


def preprocess_real_world_dnri_predictions(predictions):
    predictions = predictions[:, :, :, :, :2]  # discard velocity
    for test_case in range(40):
        for timestep in range(len(predictions[0][0])):
            predictions[test_case][0][timestep] = np.array(predictions[test_case][0][timestep])

    predictions = predictions.reshape(40, -1, 30, 2)
    length = int(predictions.shape[1])
    denormalized_prediction = []

    for i in range(40):
        denormalized_prediction.append([])
        for j in range(length):
            denormalized_prediction[i].append([])
            for k in range(NUM_PARTICLES):
                denormalized_x = (predictions[i][j][k][0] + 1) / 2
                denormalized_y = (predictions[i][j][k][1] + 1) / 2
                denormalized_prediction[i][j].append([denormalized_x, denormalized_y])

    return np.array(denormalized_prediction)


def get_error_for_real_data(model_type, seed, num_input, test_length, error_type, data_type):

    if model_type == 'static_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/real_softbody_predictions_static_{data_type}.npy')
        predictions = preprocess_real_world_dnri_predictions(predictions)

    elif model_type == 'dynamic_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/real_softbody_predictions_dynamic_{data_type}.npy')
        predictions = preprocess_real_world_dnri_predictions(predictions)

    elif model_type == 'dnri':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/real_softbody_predictions_{data_type}.npy')
        predictions = preprocess_real_world_dnri_predictions(predictions)

    elif model_type == 'dpi':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/real_softbody_predictions_{data_type}.npy')
        predictions = predictions[:, :, :, :2].tolist()

    elif model_type == 'graphrnn':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/real_softbody_predictions-num_samples_15_{data_type}.npy')
        predictions = predictions[:, :, :, :2]

    else:
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/real_softbody_predictions.npy')

    errors = []
    dnri_offset = 1
    if model_type == 'dnri':
        dnri_offset = 0

    if error_type == 'Position':
        for test_case in tqdm(REAL_DATA_EVAL_CASES):
            errors.append([])
            normalized_ground_truth_pointset = get_real_world_ground_truth_pointset(test_case)

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * REAL_DATA_OFFSET]], dtype='float32')
                errors[test_case].append(get_cd_loss(ground_truth, np.array([predictions[test_case][dnri_offset + timestep]], dtype='float32')))

    else:
        for test_case in tqdm(REAL_DATA_EVAL_CASES):
            errors.append([])
            normalized_ground_truth_pointset = get_real_world_ground_truth_pointset(test_case)

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * REAL_DATA_OFFSET]], dtype='float32')
                transformed_ground_truth = center_transform(ground_truth)
                transformed_prediction = center_transform(np.array([predictions[test_case][dnri_offset + timestep]], dtype='float32'))
                errors[test_case].append(get_cd_loss(transformed_ground_truth, transformed_prediction))

    return errors


def get_real_world_input_pointset(case, num_input, init_data_type):
    input_info, ptr = load_json(os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{case}', 'normalized_sequence_of_point_sets.json'))
    ptr.close()
    input_info = [input_info[REAL_DATA_OFFSET * i] for i in range(num_input)]

    if init_data_type == 'unordered':
        input_info = [sample(pointset, len(pointset)) for pointset in input_info]
    elif init_data_type == 'sorted_x':
        input_info = [sort_pointset_by_ascending_x(pointset) for pointset in input_info]
    elif init_data_type == 'sorted_y':
        input_info = [sort_pointset_by_descending_y(pointset) for pointset in input_info]
    else:
        print('The data type for initial input sequence for real-world data should be either unordered, sorted_x, or sorted_y')
        exit(0)

    return np.array([input_info])


def get_real_world_ground_truth_pointset(case):
    path = os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{case}', 'normalized_sequence_of_point_sets.json')
    pointsets, ptr = load_json(path)
    ptr.close()
    return pointsets


def find_case_info(path):
    directories = os.listdir(path)
    invalid_names = ['output.mp4', 'output.MP4']
    for dirname in invalid_names:
        if dirname in directories:
            directories.remove(dirname)
    num_frames = len(directories)
    directories = sorted(directories, key=lambda file: int(re.findall('\d+', file)[0]))

    first_frame_number = directories[0][9:][:-4]
    return int(first_frame_number), num_frames


def get_final_video_size(video_number):
    if video_number == 4:
        return 1624, 948
    if 42 <= video_number <= 47:
        return 1470, 926
    if video_number == 41:
        return 1480, 926
    src1 = cv2.imread(os.path.join(REAL_DATA_PATH, 'crop_positions', f'case_{video_number}_left.jpg'))
    src2 = cv2.imread(os.path.join(REAL_DATA_PATH, 'crop_positions', f'case_{video_number}_down.jpg'))
    return src1.shape[1], src2.shape[0]


def convert_to_pixel_coordinates(predicted_pointset, distance, height):
    pixel_pointset = []

    for point in predicted_pointset:
        x = int(CROP_SIZE * point[0])
        y = int(CROP_SIZE * (1 - point[1]))
        pixel_pointset.append([x, y])

    for i in range(NUM_PARTICLES):
        pixel_pointset[i][0] += 228
        pixel_pointset[i][1] += height - CROP_SIZE

    return sort_clockwise(pixel_pointset)


def sort_clockwise(pointset):

    sorted_pointset = sorted(pointset, key=lambda coord: (-coord[1], coord[0]))
    top = sorted_pointset[0]
    bottom = sorted_pointset[-1]

    right_points = []
    left_points = []
    sorted_pointset.remove(top)
    sorted_pointset.remove(bottom)
    for point in sorted_pointset:
        if point[0] >= top[0] and point[0] >= bottom[0]:
            assert bottom[1] <= point[1] <= top[1]
            right_points.append(point)
        else:
            left_points.append(point)

    right_points = sorted(right_points, key=lambda coord: (-coord[1], coord[0]))
    left_points = sorted(left_points, key=lambda coord: (coord[1], -coord[0]))
    sorted_pointset = [top]
    sorted_pointset.extend(right_points)
    sorted_pointset.append(bottom)
    sorted_pointset.extend(left_points)
    return np.array(sorted_pointset)
