import numpy as np
import os
from common.Constants import CODEC, REAL_WORLD_EVAL_CASES, REAL_DATA_PATH, NUM_PARTICLES
from evaluation.real_world_test_utils import get_real_world_ground_truth_pointset, find_case_info, get_final_video_size, convert_to_pixel_coordinates, preprocess_real_world_dnri_predictions
from common.Utils import create_directory
from compare_error_custom import OURS_SEED, DNRI_SEED, NRI_SEED, DPI_SEED, GRAPHRNN_SEED, LSTM_ORDERED_SEED, LSTM_SORTED_SEED, TP_NET_SEED
import cv2
from PIL import Image
import argparse


VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_SIZE = 860
CROP_SIZE = 922
WALL_SIZE = int((VIDEO_HEIGHT - FRAME_SIZE) / 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input', required=False, default=5)
    parser.add_argument('--data_type', required=False, default='ordered')
    parser.add_argument('--num_samples', required=False, default=15)
    parser.add_argument('--save_img', required=False, default=1)
    parser.add_argument('--offset', required=False, default=2)
    args = parser.parse_args()

    savepath = create_directory(f'../result/compare/input-{args.num_input}')

    num_models = 2
    ours_predictions = np.load(os.path.join('..', 'result', 'tp_net', f'tp_net-{args.num_input}', f'seed_{TP_NET_SEED[args.num_input - 3]}', f'real_softbody_predictions.npy'))
    #dnri_predictions = preprocess_dnri_predictions(np.load(os.path.join('..', 'result', 'dnri', f'dnri-{args.num_input}', f'seed_{DNRI_SEED[args.num_input - 3]}', f'real_softbody_predictions_{args.data_type}.npy')))
    #nri_predictions = preprocess_dnri_predictions(np.load(os.path.join('..', 'result', 'nri', f'nri-{args.num_input}', f'seed_{NRI_SEED[args.num_input - 3]}', f'real_softbody_predictions_static_{args.data_type}.npy')))
    #nri_dynamic_predictions = preprocess_dnri_predictions(np.load(os.path.join('..', 'result', 'nri', f'nri-{args.num_input}', f'seed_{NRI_SEED[args.num_input - 3]}', f'real_softbody_predictions_dynamic_{args.data_type}.npy')))
    #graphrnn_predictions = np.load(os.path.join('..', 'result', 'graphrnn', f'graphrnn-{args.num_input}', f'seed_{GRAPHRNN_SEED[args.num_input - 3]}', f'real_softbody_predictions-num_samples_{args.num_samples}_{args.data_type}.npy'))
    #dpi_predictions = np.load(os.path.join('..', 'result', 'dpi', f'dpi-{args.num_input}', f'seed_{DPI_SEED[args.num_input - 3]}', f'real_softbody_predictions_{args.data_type}.npy'))

    for num, case in enumerate(REAL_WORLD_EVAL_CASES):

        print(f'=========== Testing Case #{case} =============')
        try:
            distance, height = get_final_video_size(case)
        except:
            continue
        video_size = (575 * num_models + 40 * (num_models - 1), 520)
        # video_size = ((228 + CROP_SIZE) * 2 + 50, 1040)
        # 1150 * 2 + 50
        # 1040
        output_video = cv2.VideoWriter(os.path.join(savepath, f'Real Test Case_{case}.MP4'), CODEC, 30, video_size)

        if args.save_img:
            frames_savepath = create_directory(os.path.join(savepath, 'real_images', f'Real Test Case {case}'))
        background_images = [cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)] for _ in range(num_models)]

        ground_truth_base_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{case}')
        ground_truth_pointsets = get_real_world_ground_truth_pointset(case, test_data_type='ordered')
        test_length = int(len(ground_truth_pointsets) / args.offset) - args.num_input - 1

        first_frame_number, num_frames = find_case_info(ground_truth_base_path)

        timestep = 0
        iteration = 0

        while iteration < args.num_input + 1:
            merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
            ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))
            merged.paste(im=ground_truth_image, box=(0, 0))
            merged.paste(im=ground_truth_image, box=(575 + 40, 0))
            #merged.paste(im=ground_truth_image, box=(575 * 2 + 80, 0))
            #merged.paste(im=ground_truth_image, box=(575 * 3 + 120, 0))
            #merged.paste(im=ground_truth_image, box=(575 * 4 + 160, 0))
            #merged.paste(im=ground_truth_image, box=(575 * 5 + 200, 0))

            merged = np.array(merged)
            output_video.write(merged)
            if args.save_img:
                cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)

            timestep += args.offset
            iteration += 1

        iteration = 0
        while iteration < test_length:

            #ground_truth_pointset = ground_truth_pointsets[timestep]
            #gt_pixel_coordinates = convert_to_pixel_coordinates(ground_truth_pointset, distance, height)
            #cv2.fillPoly(img=background_images[0], pts=[gt_pixel_coordinates], color=(47, 164, 193))
            #for i in range(NUM_PARTICLES):
            #    background_images[0] = cv2.line(background_images[0], gt_pixel_coordinates[i], gt_pixel_coordinates[i], (47, 164, 193), thickness=15)


            ours_pointset = ours_predictions[num][1 + iteration]
            ours_pixel_coordinates = convert_to_pixel_coordinates(ours_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[1] = cv2.line(background_images[1], ours_pixel_coordinates[i], ours_pixel_coordinates[i], (47, 164, 193), thickness=12)
            """
            dpi_pointset = dpi_predictions[num][1 + iteration]
            dpi_pixel_coordinates = convert_to_pixel_coordinates(dpi_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[2] = cv2.line(background_images[2], dpi_pixel_coordinates[i], dpi_pixel_coordinates[i], (47, 164, 193), thickness=12)

            dnri_pointset = dnri_predictions[num][iteration]
            dnri_pixel_coordinates = convert_to_pixel_coordinates(dnri_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[3] = cv2.line(background_images[3], dnri_pixel_coordinates[i], dnri_pixel_coordinates[i], (47, 164, 193), thickness=12)

            static_nri_pointset = nri_predictions[num][1 + iteration]
            static_nri_pixel_coordinates = convert_to_pixel_coordinates(static_nri_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[4] = cv2.line(background_images[4], static_nri_pixel_coordinates[i], static_nri_pixel_coordinates[i], (47, 164, 193), thickness=12)

            dyanmic_nri_pointset = nri_dynamic_predictions[num][1 + iteration]
            dyanmic_nri_pixel_coordinates = convert_to_pixel_coordinates(dyanmic_nri_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[5] = cv2.line(background_images[5], dyanmic_nri_pixel_coordinates[i], dyanmic_nri_pixel_coordinates[i], (47, 164, 193), thickness=12)
            
            graphrnn_pointset = graphrnn_predictions[num][1 + iteration]
            graphrnn_pixel_coordinates = convert_to_pixel_coordinates(graphrnn_pointset, distance, height)
            for i in range(NUM_PARTICLES):
                background_images[1] = cv2.line(background_images[1], graphrnn_pixel_coordinates[i], graphrnn_pixel_coordinates[i], (47, 164, 193), thickness=12)
            """

            ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

            merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
            merged.paste(im=ground_truth_image, box=(0, 0))
            merged.paste(im=Image.fromarray(cv2.resize(background_images[1], dsize=(575, 520))), box=(575 + 40, 0))
            #merged.paste(im=Image.fromarray(cv2.resize(background_images[2], dsize=(575, 520))), box=(575 * 2 + 80, 0))
            #merged.paste(im=Image.fromarray(cv2.resize(background_images[3], dsize=(575, 520))), box=(575 * 3 + 120, 0))
            #merged.paste(im=Image.fromarray(cv2.resize(background_images[4], dsize=(575, 520))), box=(575 * 4 + 160, 0))
            #merged.paste(im=Image.fromarray(cv2.resize(background_images[5], dsize=(575, 520))), box=(575 * 5 + 200, 0))
            merged = np.array(merged)

            output_video.write(merged)
            if args.save_img:
                cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)

            del background_images
            background_images = [cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)] for i in range(num_models)]

            timestep += args.offset
            iteration += 1

        output_video.release()
        cv2.destroyAllWindows()
