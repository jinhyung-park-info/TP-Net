import numpy as np
import os
from common.Constants import DNRI_TEST_CASES, CODEC, VIDEO_HEIGHT, WALL_SIZE
from simulation_test_utils import get_ground_truth_pointset, draw_box2d_image
from common.Utils import denormalize_dnri_pointset, get_nested_pointset, denormalize_pointset
from tqdm import tqdm
import cv2
from PIL import Image

TEST_LENGTH = 150
TEST_ANIMATIONS = 20
offset = 4
start_timestep = 0
predict_start_timestep = 3

timestamps = [i for i in range(TEST_LENGTH)]
dnri_predictions = np.load('./softbody_predictions_dnri.npy')
nri_predictions = np.load('./softbody_predictions_nri.npy')
nri_dynamic_predictions = np.load('./softbody_predictions_nri_dynamic.npy')
ours_predictions = np.load(os.path.join(f'../result/global_pointnet/version_13/simulation_prediction_result/offset_4_test_cases/softbody_predictions_ours_start_{start_timestep}.npy'))

for test_case_num, test_case in enumerate(DNRI_TEST_CASES[:TEST_ANIMATIONS]):

    ground_truth_pointsets = get_ground_truth_pointset(test_case)
    output_video = cv2.VideoWriter(os.path.join(f'Test Case_{test_case_num}.MP4'), CODEC, 30, (VIDEO_HEIGHT * 5, VIDEO_HEIGHT))
    timestep = 0
    iteration = 0

    while iteration < predict_start_timestep:
        merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 5, VIDEO_HEIGHT), color=255)
        ground_truth_image = Image.fromarray(draw_box2d_image(ground_truth_pointsets[timestep]))
        merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))
        merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
        merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
        merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
        merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 4 + WALL_SIZE, WALL_SIZE))
        output_video.write(np.array(merged))
        timestep += 4
        iteration += 1

    iteration = 0
    while iteration < TEST_LENGTH:
        merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 5, VIDEO_HEIGHT), color=255)
        merged.paste(im=Image.fromarray(draw_box2d_image(ground_truth_pointsets[timestep])), box=(WALL_SIZE, WALL_SIZE))
        merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_pointset(ours_predictions[test_case_num][iteration]))), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
        merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(dnri_predictions[test_case_num][0][iteration][:, :2]))), box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
        merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(nri_predictions[test_case_num][0][iteration][:, :2]))), box=(VIDEO_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
        merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(nri_dynamic_predictions[test_case_num][0][iteration][:, :2]))), box=(VIDEO_HEIGHT * 4 + WALL_SIZE, WALL_SIZE))

        output_video.write(np.array(merged))
        timestep += 4
        iteration += 1

    output_video.release()
    cv2.destroyAllWindows()
