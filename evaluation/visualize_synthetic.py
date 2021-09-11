import numpy as np
import os
from common.Constants import SIM_DATA_EVAL_CASES, SIM_DATA_OFFSET
from evaluation.simulation_test_utils import get_ground_truth_pointset, visualize_synthetic_pointset
from common.Utils import denormalize_pointset, create_directory
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

# Constants - Do not change this
VIDEO_HEIGHT = 900
FRAME_SIZE = 860
WALL_SIZE = int((VIDEO_HEIGHT - FRAME_SIZE) / 2)
CODEC = cv2.VideoWriter_fourcc(*'MPV4')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input', required=False, default=4)
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--init_data_type', required=False, default='ordered')
    parser.add_argument('--video_length', required=False, default=150)
    parser.add_argument('--test_animations', required=False, default=60)
    args = parser.parse_args()

    result_path = os.path.join('..', 'result', 'tp_net', f'tp_net-{args.num_input}', f'seed_{args.seed}')
    savepath = create_directory(os.path.join(result_path, 'rollout_video', 'synthetic_dataset'))
    test_cases = SIM_DATA_EVAL_CASES[:args.test_animations]
    timestamps = [i for i in range(args.video_length)]

    predicted_pointsets = np.load(os.path.join(result_path, f'softbody_predictions_{args.init_data_type}.npy'))

    for test_case_num, test_case in enumerate(test_cases):

        print(f'=========== Visualizing Test Case #{test_case_num} =============')

        ground_truth_pointsets = get_ground_truth_pointset(test_case)
        output_video = cv2.VideoWriter(os.path.join(savepath, f'Test Case_{test_case_num}.MP4'), CODEC, 30, (VIDEO_HEIGHT * 2, VIDEO_HEIGHT))

        timestep = 0
        iteration = 0

        while iteration < args.num_input + 1:
            merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 2, VIDEO_HEIGHT), color=255)
            ground_truth_image = Image.fromarray(visualize_synthetic_pointset(ground_truth_pointsets[timestep], FRAME_SIZE, WALL_SIZE))
            merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))
            merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
            output_video.write(np.array(merged))
            timestep += SIM_DATA_OFFSET
            iteration += 1

        iteration = 0
        while iteration < args.video_length:
            merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 2, VIDEO_HEIGHT), color=255)
            merged.paste(im=Image.fromarray(visualize_synthetic_pointset(ground_truth_pointsets[timestep], FRAME_SIZE, WALL_SIZE)), box=(WALL_SIZE, WALL_SIZE))
            merged.paste(im=Image.fromarray(visualize_synthetic_pointset(denormalize_pointset(predicted_pointsets[test_case_num][1 + iteration]), FRAME_SIZE, WALL_SIZE)), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
            output_video.write(np.array(merged))
            timestep += SIM_DATA_OFFSET
            iteration += 1

        output_video.release()
        cv2.destroyAllWindows()
