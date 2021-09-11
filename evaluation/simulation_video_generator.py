import numpy as np
import os
from common.Constants import EVAL_CASES, CODEC, VIDEO_HEIGHT, WALL_SIZE, REAL_WORLD_TEST_CASES
from evaluation.simulation_test_utils import get_ground_truth_pointset, draw_box2d_image
from evaluation.real_world_test_utils import get_real_world_ground_truth_pointset
from common.Utils import denormalize_dnri_pointset, denormalize_pointset, create_directory
from compare_error_custom import OURS_SEED, DNRI_SEED, NRI_SEED, DPI_SEED, GRAPHRNN_SEED, LSTM_ORDERED_SEED, LSTM_SORTED_SEED, TP_NET_SEED
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

# 6. Model Related
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
SMALL_HEIGHT = 450
FRAME_SIZE = 860
SMALL_SIZE = 430
WALL_SIZE = int((VIDEO_HEIGHT - FRAME_SIZE) / 2)
CODEC = cv2.VideoWriter_fourcc(*'MPV4')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input', required=False, default=5)
    parser.add_argument('--num_samples', required=False, default=15)
    parser.add_argument('--save_img', required=False, default=1)
    parser.add_argument('--test_length', required=False, default=150)
    parser.add_argument('--test_animations', required=False, default=25)
    parser.add_argument('--offset', required=False, default=4)
    args = parser.parse_args()

    savepath = create_directory(f'../result/compare/input-5-graphrnn')
    test_cases = EVAL_CASES[:args.test_animations]

    timestamps = [i for i in range(args.test_length)]

    #tp_net_predictions = np.load(os.path.join('..', 'result', 'tp_net', f'tp_net-{args.num_input}', f'seed_{TP_NET_SEED[args.num_input - 3]}', f'softbody_predictions.npy'))
    #ours_predictions = np.load(os.path.join('..', 'result', 'global_pointnet', f'global_pointnet-{args.num_input}', f'seed_{OURS_SEED[args.num_input - 3]}', f'softbody_predictions.npy'))
    #dnri_predictions = np.load(os.path.join('..', 'result', 'dnri', f'dnri-{args.num_input}', f'seed_{DNRI_SEED[args.num_input - 3]}', f'softbody_predictions.npy'))
    #nri_predictions = np.load(os.path.join('..', 'result', 'nri', f'nri-{args.num_input}', f'seed_{NRI_SEED[args.num_input - 3]}', f'softbody_predictions_static.npy'))
    #nri_dynamic_predictions = np.load(os.path.join('..', 'result', 'nri', f'nri-{args.num_input}', f'seed_{NRI_SEED[args.num_input - 3]}', f'softbody_predictions_dynamic.npy'))
    graphrnn_predictions = np.load(os.path.join('..', 'result', 'graphrnn', f'graphrnn-{args.num_input}', f'seed_{GRAPHRNN_SEED[args.num_input - 3]}', f'softbody_predictions-num_samples_{args.num_samples}.npy'))
    #dpi_predictions = np.load(os.path.join('..', 'result', 'dpi', f'dpi-{args.num_input}', f'seed_{DPI_SEED[args.num_input - 3]}', f'softbody_predictions.npy'))

    for test_case_num, test_case in tqdm(list(enumerate(test_cases))):
        if args.save_img:
            img_save_path = create_directory(f'{savepath}/images/Test Case_{test_case_num}')

        if test_case_num != 9:
            continue

        ground_truth_pointsets = get_ground_truth_pointset(test_case)

        output_video = cv2.VideoWriter(os.path.join(savepath, f'Test Case_{test_case_num}.MP4'), CODEC, 30, (VIDEO_HEIGHT * 2, VIDEO_HEIGHT))

        timestep = 0
        iteration = 0

        while iteration < args.num_input + 1:
            merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 2, VIDEO_HEIGHT), color=255)
            ground_truth_image = Image.fromarray(draw_box2d_image(ground_truth_pointsets[timestep]))
            merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))
            merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 4 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 5 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT * 6 + WALL_SIZE, WALL_SIZE))
            output_video.write(np.array(merged))
            timestep += 4
            iteration += 1
            if args.save_img:
                merged.save(os.path.join(img_save_path, f'Timestep_{timestep}.jpg'))

        iteration = 0
        while iteration < args.test_length:
            merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 2, VIDEO_HEIGHT), color=255)
            merged.paste(im=Image.fromarray(draw_box2d_image(ground_truth_pointsets[timestep])), box=(WALL_SIZE, WALL_SIZE))
            #merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_pointset(tp_net_predictions[test_case_num][1 + iteration]))), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_pointset(ours_predictions[test_case_num][1 + iteration]))), box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(dnri_predictions[test_case_num][0][iteration][:, :2]))), box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(nri_predictions[test_case_num][0][1 + iteration][:, :2]))), box=(VIDEO_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
            #merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_dnri_pointset(nri_dynamic_predictions[test_case_num][0][1 + iteration][:, :2]))), box=(VIDEO_HEIGHT * 4 + WALL_SIZE, WALL_SIZE))
            merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_pointset(graphrnn_predictions[test_case_num][1 + iteration]))), box=(VIDEO_HEIGHT * 1 + WALL_SIZE, WALL_SIZE))
            #try:
            #    merged.paste(im=Image.fromarray(draw_box2d_image(denormalize_pointset(dpi_predictions[test_case_num][1 + iteration][:, :2].tolist()))), box=(VIDEO_HEIGHT * 6 + WALL_SIZE, WALL_SIZE))
            #except:
            #    pass
            output_video.write(np.array(merged))
            timestep += 4
            iteration += 1
            if args.save_img:
                merged.save(os.path.join(img_save_path, f'Timestep_{timestep}.jpg'))

        output_video.release()
        cv2.destroyAllWindows()
