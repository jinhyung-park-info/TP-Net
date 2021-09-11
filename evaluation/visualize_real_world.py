import numpy as np
import os
from common.Constants import REAL_DATA_PATH, NUM_PARTICLES, REAL_DATA_OFFSET
from evaluation.real_world_test_utils import get_real_world_ground_truth_pointset, convert_to_pixel_coordinates
from common.Utils import create_directory
import cv2
from PIL import Image
import argparse

# Constants - Do not change this
VIDEO_SIZE = (575 * 2 + 40, 520)
CROP_SIZE = 922

# Due to file size limit, we only provide real-world image frames for only one real-world test case with low resolution
# The constants below are specific constants hard-coded for the sample test case
SAMPLE_TEST_CASE = 24
first_frame_number = 78
distance, height = (1494, 940)

CODEC = cv2.VideoWriter_fourcc(*'MPV4')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input', required=False, default=3)
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--init_data_type', required=False, default='ordered')
    args = parser.parse_args()

    result_path = os.path.join('..', 'result', 'tp_net', f'tp_net-{args.num_input}', f'seed_{args.seed}')
    savepath = create_directory(os.path.join(result_path, 'rollout_video', 'real_world_dataset'))
    predicted_pointsets = np.load(os.path.join(result_path, f'real_softbody_predictions_{args.init_data_type}.npy'))

    print(f'=========== Visualizing Sample Test Case #{SAMPLE_TEST_CASE} =============')

    output_video = cv2.VideoWriter(os.path.join(savepath, f'Test Case_{SAMPLE_TEST_CASE}.MP4'), CODEC, 30, VIDEO_SIZE)
    background_image = cv2.imread(os.path.join(REAL_DATA_PATH, f'case_{SAMPLE_TEST_CASE}', 'original_frames', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
    ground_truth_pointsets = get_real_world_ground_truth_pointset(SAMPLE_TEST_CASE)
    test_length = int(len(ground_truth_pointsets) / args.offset) - args.num_input - 1

    timestep = 0
    iteration = 0

    while iteration < args.num_input + 1:
        merged = Image.new(mode="RGB", size=VIDEO_SIZE, color=(255, 255, 255))
        ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, f'case_{SAMPLE_TEST_CASE}', 'original_frames', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
        ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))
        merged.paste(im=ground_truth_image, box=(0, 0))
        merged.paste(im=ground_truth_image, box=(575 + 40, 0))
        merged = np.array(merged)
        output_video.write(merged)

        timestep += REAL_DATA_OFFSET
        iteration += 1

    iteration = 0
    while iteration < test_length:
        ours_pointset = predicted_pointsets[SAMPLE_TEST_CASE][1 + iteration]
        ours_pixel_coordinates = convert_to_pixel_coordinates(ours_pointset, height, CROP_SIZE)
        for i in range(NUM_PARTICLES):
            background_image = cv2.line(background_image, ours_pixel_coordinates[i], ours_pixel_coordinates[i], (47, 164, 193), thickness=12)

        ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, f'case_{SAMPLE_TEST_CASE}', 'original_frames', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
        ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))
        merged = Image.new(mode="RGB", size=VIDEO_SIZE, color=(255, 255, 255))
        merged.paste(im=ground_truth_image, box=(0, 0))
        merged.paste(im=Image.fromarray(cv2.resize(background_image, dsize=(575, 520))), box=(575 + 40, 0))
        merged = np.array(merged)
        output_video.write(merged)

        del background_image
        background_image = cv2.imread(os.path.join(REAL_DATA_PATH, f'case_{SAMPLE_TEST_CASE}', 'original_frames', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]

        timestep += REAL_DATA_OFFSET
        iteration += 1

    output_video.release()
    cv2.destroyAllWindows()
