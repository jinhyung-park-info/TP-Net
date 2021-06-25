import numpy as np
import os
from common.Constants import DNRI_TEST_CASES
from simulation_test_utils import get_ground_truth_pointset, get_area_loss
from loss import get_cd_loss_func
from common.Utils import denormalize_dnri_pointset, normalize_nested_pointset, normalize_pointset, create_directory
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_dnri_predictions(predictions):
    predictions = predictions[:, :, :TEST_LENGTH, :, :2]  # discard velocity
    for test_case in range(100):
        for timestep in range(TEST_LENGTH):
            predictions[test_case][0][timestep] = np.array(normalize_nested_pointset(denormalize_dnri_pointset(predictions[test_case][0][timestep])))

    return predictions.reshape(100, TEST_LENGTH, 30, 2)


TEST_LENGTH = 150
TEST_ANIMATIONS = 100
offset = 4
start_timestep = 0
predict_start_timestep = 3
include_shape_error = False
include_baselines = False
timestamps = [i for i in range(TEST_LENGTH)]
dnri_predictions = np.load('./softbody_predictions_dnri.npy')
dnri_predictions = preprocess_dnri_predictions(dnri_predictions)

nri_predictions = np.load('./softbody_predictions_nri.npy')
nri_predictions = preprocess_dnri_predictions(nri_predictions)

nri_dynamic_predictions = np.load('./softbody_predictions_nri_dynamic.npy')
nri_dynamic_predictions = preprocess_dnri_predictions(nri_dynamic_predictions)

ours_predictions = np.load(os.path.join(f'../result/global_pointnet/version_13/simulation_prediction_result/offset_4_test_cases/softbody_predictions_ours_start_{start_timestep}.npy'))

if include_baselines:
    sorted_predictions = np.load(os.path.join(f'../result/lstm/version_45/simulation_prediction_result/offset_4_test_cases/softbody_predictions_ours_start_{start_timestep}.npy'))
    ordered_predictions = np.load(os.path.join(f'../result/lstm/version_44/simulation_prediction_result/offset_4_test_cases/softbody_predictions_ours_start_{start_timestep}.npy'))


#print(dnri_predictions.shape) #== (100, 150, 30, 2)
#print(ours_predictions.shape) #== (100, 150, 30, 2)

ours_position_errors_all = np.array([0.0] * TEST_LENGTH)
if include_baselines:
    ordered_position_errors_all = np.array([0.0] * TEST_LENGTH)
    sorted_position_errors_all = np.array([0.0] * TEST_LENGTH)
dnri_position_errors_all = np.array([0.0] * TEST_LENGTH)
nri_position_errors_all = np.array([0.0] * TEST_LENGTH)
nri_dynamic_position_errors_all = np.array([0.0] * TEST_LENGTH)

if include_shape_error:
    ours_shape_errors_all = np.array([0.0] * TEST_LENGTH)
    if include_baselines:
        ordered_shape_errors_all = np.array([0.0] * TEST_LENGTH)
        sorted_shape_errors_all = np.array([0.0] * TEST_LENGTH)
    dnri_shape_errors_all = np.array([0.0] * TEST_LENGTH)
    nri_shape_errors_all = np.array([0.0] * TEST_LENGTH)
    nri_dynamic_shape_errors_all = np.array([0.0] * TEST_LENGTH)

for test_case_num, test_case in tqdm(enumerate(DNRI_TEST_CASES[:TEST_ANIMATIONS])):

    ground_truth_pointsets = get_ground_truth_pointset(test_case)
    normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

    ours_position_errors = []
    if include_baselines:
        ordered_position_errors = []
        sorted_position_errors = []
    dnri_position_errors = []
    nri_position_errors = []
    nri_dynamic_position_errors = []

    if include_shape_error:
        ours_shape_errors = []
        ordered_shape_errors = []
        sorted_shape_errors = []
        dnri_shape_errors = []
        nri_shape_errors = []
        nri_dynamic_shape_errors = []

    for timestep in range(TEST_LENGTH):

        if include_shape_error:
            ours_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), ours_predictions[test_case_num][timestep]))
            ordered_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), ordered_predictions[test_case_num][timestep]))
            sorted_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), sorted_predictions[test_case_num][timestep]))
            dnri_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), dnri_predictions[test_case_num][timestep]))
            nri_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), nri_predictions[test_case_num][timestep]))
            nri_dynamic_shape_errors.append(get_area_loss(np.array(normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset], dtype='float32'), nri_dynamic_predictions[test_case_num][timestep]))

        ours_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([ours_predictions[test_case_num][timestep]], dtype='float32')))
        if include_baselines:
            ordered_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([ordered_predictions[test_case_num][timestep]], dtype='float32')))
            sorted_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([sorted_predictions[test_case_num][timestep]], dtype='float32')))
        dnri_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([dnri_predictions[test_case_num][timestep]], dtype='float32')))
        nri_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([nri_predictions[test_case_num][timestep]], dtype='float32')))
        nri_dynamic_position_errors.append(get_cd_loss_func(np.array([normalized_ground_truth_pointset[predict_start_timestep * offset + timestep * offset]], dtype='float32'), np.array([nri_dynamic_predictions[test_case_num][timestep]], dtype='float32')))

    ours_position_errors_all += np.array(ours_position_errors)
    if include_baselines:
        ordered_position_errors_all += np.array(ordered_position_errors)
        sorted_position_errors_all += np.array(sorted_position_errors)
    dnri_position_errors_all += np.array(dnri_position_errors)
    nri_position_errors_all += np.array(nri_position_errors)
    nri_dynamic_position_errors_all += np.array(nri_dynamic_position_errors)

    if include_shape_error:
        ours_shape_errors_all += np.array(ours_shape_errors)
        ordered_shape_errors_all += np.array(ordered_shape_errors)
        sorted_shape_errors_all += np.array(sorted_shape_errors)
        dnri_shape_errors_all += np.array(dnri_shape_errors)
        nri_shape_errors_all += np.array(nri_shape_errors)
        nri_dynamic_shape_errors_all += np.array(nri_dynamic_shape_errors)

    # Position Error
    plt.plot(timestamps, ours_position_errors, label=f'Ours')
    if include_baselines:
        plt.plot(timestamps, ordered_position_errors, label=f'Baseline (Ordered)')
        plt.plot(timestamps, sorted_position_errors, label=f'Baseline (Sorted)')
    plt.plot(timestamps, dnri_position_errors, label=f'dNRI')
    plt.plot(timestamps, nri_position_errors, label=f'NRI')
    plt.plot(timestamps, nri_dynamic_position_errors, label=f'NRI_Dynamic')

    plt.xlabel('Timestep')
    plt.ylabel('Position Error')
    plt.legend()
    plt.savefig(os.path.join(create_directory('position_error'), 'Test Case {}.png'.format(test_case_num)), dpi=600)
    plt.clf()

    if include_shape_error:
        # Shape Error
        plt.plot(timestamps, ours_shape_errors, label=f'Ours')
        plt.plot(timestamps, ordered_shape_errors, label=f'Baseline (Ordered)')
        plt.plot(timestamps, sorted_shape_errors, label=f'Baseline (Sorted)')
        plt.plot(timestamps, dnri_shape_errors, label=f'dNRI')
        plt.plot(timestamps, nri_shape_errors, label=f'NRI')
        plt.plot(timestamps, nri_dynamic_shape_errors, label=f'NRI_Dynamic')

        plt.xlabel('Timestep')
        plt.ylabel('Shape Error')
        plt.legend()
        plt.savefig(os.path.join(create_directory('shape_error'), 'Test Case {}.png'.format(test_case_num)), dpi=600)
        plt.clf()


# Average Position Error Graph
ours_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
if include_baselines:
    ordered_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
    sorted_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
dnri_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
nri_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
nri_dynamic_position_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])

plt.plot(timestamps, ours_position_errors_all.tolist(), label=f'Ours')
if include_baselines:
    plt.plot(timestamps, ordered_position_errors_all.tolist(), label=f'Baseline (Ordered)')
    plt.plot(timestamps, sorted_position_errors_all.tolist(), label=f'Baseline (Sorted)')
plt.plot(timestamps, dnri_position_errors_all.tolist(), label=f'dNRI')
plt.plot(timestamps, nri_position_errors_all.tolist(), label=f'NRI')
plt.plot(timestamps, nri_dynamic_position_errors_all.tolist(), label=f'NRI_Dynamic')

plt.xlabel('Timestep')
plt.ylabel('Average Position Error')
plt.legend()
plt.savefig(os.path.join('.', 'Average Position Error.png'), dpi=600)
plt.clf()


if include_shape_error:
    # Average Shape Error Graph
    ours_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
    if include_baselines:
        ordered_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
        sorted_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
    dnri_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
    nri_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])
    nri_dynamic_shape_errors_all /= len(DNRI_TEST_CASES[:TEST_ANIMATIONS])

    plt.plot(timestamps, ours_shape_errors_all.tolist(), label=f'Ours')
    if include_baselines:
        plt.plot(timestamps, ordered_shape_errors_all.tolist(), label=f'Baseline (Ordered)')
        plt.plot(timestamps, sorted_shape_errors_all.tolist(), label=f'Baseline (Sorted)')
    plt.plot(timestamps, dnri_shape_errors_all.tolist(), label=f'dNRI')
    plt.plot(timestamps, nri_shape_errors_all.tolist(), label=f'NRI')
    plt.plot(timestamps, nri_dynamic_shape_errors_all.tolist(), label=f'NRI_Dynamic')

    plt.xlabel('Timestep')
    plt.ylabel('Average Shape Error')
    plt.legend()
    plt.savefig(os.path.join('.', 'Average Shape Error.png'), dpi=600)
    plt.clf()
