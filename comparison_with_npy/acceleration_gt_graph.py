import numpy as np
from common.Constants import DNRI_TEST_CASES
from simulation_test_utils import get_ground_truth_pointset
from common.Utils import normalize_nested_pointset, denormalize_dnri_pointset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

TEST_LENGTH = 150
TEST_ANIMATIONS = 20
PARTICLE_NUM = 0
offset = 4
start_timestep = 0
timesteps = [i for i in range(2, TEST_LENGTH)]
predicted_timesteps = [i for i in range(5, TEST_LENGTH)]
nri_predicted_timesteps = [i for i in range(42, TEST_LENGTH)]


def calculate_acceleration(timestep, pointset):
    velocity_1 = np.array(pointset[(timestep - 2)][PARTICLE_NUM]) - np.array(pointset[(timestep - 1)][PARTICLE_NUM])
    velocity_2 = np.array(pointset[timestep][PARTICLE_NUM]) - np.array(pointset[(timestep - 1)][PARTICLE_NUM])
    return velocity_2 - velocity_1


def calculate_predicted_acceleration(timestep, pointset):
    velocity_1 = np.array(pointset[timestep - 2][PARTICLE_NUM]) - np.array(pointset[timestep - 1][PARTICLE_NUM])
    velocity_2 = np.array(pointset[timestep][PARTICLE_NUM]) - np.array(pointset[timestep - 1][PARTICLE_NUM])
    return velocity_2 - velocity_1


def preprocess_dnri_predictions(predictions):
    predictions = predictions[:, :, :TEST_LENGTH, :, :2]  # discard velocity
    for test_case in range(100):
        for timestep in range(TEST_LENGTH):
            predictions[test_case][0][timestep] = np.array(normalize_nested_pointset(denormalize_dnri_pointset(predictions[test_case][0][timestep])))

    return predictions.reshape(100, TEST_LENGTH, 30, 2)


ours_predictions = np.load(os.path.join(f'../result/global_pointnet/version_13/simulation_prediction_result/offset_4_test_cases/softbody_predictions_ours_start_{start_timestep}.npy'))

dnri_predictions = np.load('./softbody_predictions_dnri.npy')
dnri_predictions = preprocess_dnri_predictions(dnri_predictions)

nri_predictions = np.load('./softbody_predictions_nri.npy')
nri_predictions = preprocess_dnri_predictions(nri_predictions)

for test_case_num, test_case in tqdm(enumerate(DNRI_TEST_CASES[:TEST_ANIMATIONS])):

    x_accelerations = []
    y_accelerations = []
    ours_x_predicted_accelerations = []
    ours_y_predicted_accelerations = []
    dnri_x_predicted_accelerations = []
    dnri_y_predicted_accelerations = []
    nri_x_predicted_accelerations = []
    nri_y_predicted_accelerations = []

    ground_truth_pointsets = get_ground_truth_pointset(test_case)
    normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

    for i in range(2, TEST_LENGTH):
        ground_truth_acceleration = calculate_acceleration(i, normalized_ground_truth_pointset)
        x_accelerations.append(ground_truth_acceleration[0])
        y_accelerations.append(ground_truth_acceleration[1])

    for i in range(2, TEST_LENGTH - 3):
        predicted_acceleration = calculate_predicted_acceleration(i, ours_predictions[test_case_num])
        ours_x_predicted_accelerations.append(predicted_acceleration[0])
        ours_y_predicted_accelerations.append(predicted_acceleration[1])

    for i in range(2, TEST_LENGTH - 40):
        dnri_predicted_acceleration = calculate_predicted_acceleration(i, dnri_predictions[test_case_num])
        dnri_x_predicted_accelerations.append(dnri_predicted_acceleration[0])
        dnri_y_predicted_accelerations.append(dnri_predicted_acceleration[1])

        nri_predicted_acceleration = calculate_predicted_acceleration(i, nri_predictions[test_case_num])
        nri_x_predicted_accelerations.append(nri_predicted_acceleration[0])
        nri_y_predicted_accelerations.append(nri_predicted_acceleration[1])

    plt.clf()
    plt.plot(timesteps, x_accelerations, label='Ground Truth')
    plt.xlabel('Timestep')
    plt.ylabel('X Acceleration')
    plt.legend()
    plt.savefig(f'x_acceleration_{test_case_num}_gt.png', dpi=600)

    plt.clf()
    plt.plot(timesteps, y_accelerations, label='Ground Truth')
    plt.xlabel('Timestep')
    plt.ylabel('Y Acceleration')
    plt.legend()
    plt.savefig(f'y_acceleration_{test_case_num}_gt.png', dpi=600)
