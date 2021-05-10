from common.Utils import create_directory, write_json, load_json, sort_pointset, shuffle_pointset, get_nested_pointset
from common.Constants import *
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import shutil
import math

CROP_SIZE = 922
CRITICAL_LENGTH_IN_SECONDS = 30


def video_to_image(video_numbers, start_times):

    base_path = os.path.join(REAL_DATA_PATH, '01_original_videos')

    for video_number, start_time in tqdm(zip(video_numbers, start_times)):

        print(f'========== Processing Video {video_number} =========')

        video_path = os.path.join(base_path, f'case_{video_number}.mp4')
        frame_savepath = create_directory(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{video_number}'))

        video = cv2.VideoCapture(video_path)
        start_frame = start_time * 30
        end_frame = start_frame + CRITICAL_LENGTH_IN_SECONDS * 30

        count = 0
        critical_frame_count = 0

        while video.isOpened():
            ret, frame = video.read()

            if frame is None or count >= end_frame:
                break
            if start_frame < count < end_frame:
                cv2.imwrite(f'{frame_savepath}/timestep_{critical_frame_count}.jpg', frame)
                critical_frame_count += 1

            count += 1

            if count % 500 == 0:
                print(f'Counted {count} frames for video {video_number}. Start Frame: {start_frame}, End Frame: {end_frame}')

        video.release()


def image_background_subtraction(video_numbers, algorithm):

    for video_number in video_numbers:
        print(f'============= Processing Video {video_number} ============')
        image_set_path = os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{video_number}')
        savepath = create_directory(os.path.join(REAL_DATA_PATH, '03_critical_frames_subtracted', f'case_{video_number}'))

        if algorithm == 'MOG2':
            background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            background_subtractor.setShadowValue(0)
        else:
            background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
            background_subtractor.setShadowValue(0)


        paths = os.listdir(image_set_path)
        paths = sorted(paths, key=lambda file: int(re.findall('\d+', file)[0]))

        for filename in tqdm(paths):
            frame = cv2.imread(os.path.join(image_set_path, filename))

            subtracted_frame = background_subtractor.apply(frame)
            cv2.imwrite(os.path.join(savepath, filename), subtracted_frame)


def get_height_distance(video_number):
    src1 = cv2.imread(os.path.join(f'C:/Users/User/Desktop/crop_positions/case_{video_number}_left.jpg'))
    src2 = cv2.imread(os.path.join(f'C:/Users/User/Desktop/crop_positions/case_{video_number}_down.jpg'))
    return src1.shape[1], src2.shape[0]


def crop_images(video_numbers):

    for video_number in tqdm(video_numbers):
        distance, height = get_height_distance(video_number)

        frames_path = os.path.join(REAL_DATA_PATH, '03_critical_frames_subtracted', f'case_{video_number}')
        filenames = os.listdir(frames_path)
        savepath = create_directory(os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{video_number}'))

        for fname in filenames:
            src = cv2.imread(os.path.join(frames_path, fname))
            dst = src[height - CROP_SIZE:height, 1920 - distance:1920 - distance + CROP_SIZE]
            cv2.imwrite(os.path.join(savepath, fname), dst)

        #image_to_video(frames_paths=[os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{video_number}')], fps=30)


def compute_closest_point(normalized_center_x, normalized_center_y, normalized_xs, normalized_ys, k):
    if k == 15:
        # y = normalized_center_y
        # ax + by + c = 0 : y - normalized_center_y = 0
        a = 0
        b = 1
        c = - normalized_center_y

    else:
        a = math.tan((2 * math.pi / NUM_PARTICLES) * k)
        b = -1
        c = - normalized_center_x * math.tan((2 * math.pi / NUM_PARTICLES) * k) + normalized_center_y

    min_distance = 10
    closest_x = None
    closest_y = None

    for x1, y1 in zip(normalized_xs, normalized_ys):
        distance = abs(a * x1 + b * y1 + c) / ((a**2 + b**2)**0.5)
        if distance < min_distance:
            if k == 0:
                if x1 > normalized_center_x:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            elif 1 <= k <= 7: # first quadrant
                if y1 > normalized_center_y and x1 > normalized_center_x:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            elif 8 <= k <= 14: # second quadrant
                if y1 > normalized_center_y and x1 < normalized_center_x:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            elif k == 15:    # horizontal left line
                if x1 < normalized_center_x:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            elif 16 <= k <= 22:  # third quadrant
                if x1 < normalized_center_x and y1 < normalized_center_y:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            elif 23 <= k <= 29:  # fourth quadrant
                if x1 > normalized_center_x and y1 < normalized_center_y:
                    closest_x = x1
                    closest_y = y1
                    min_distance = distance

            else:
                print("Should Not Reach Here")
                exit(1)

    return [closest_x, closest_y]


def get_point_cloud_data(video_numbers):

    for video_number in video_numbers:
        images_directory = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{video_number}')
        savepath = create_directory(os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{video_number}'))
        tmp_savepath = create_directory(os.path.join(savepath, 'tmp'))
        pointsets = []

        filenames = os.listdir(images_directory)
        if 'output.mp4' in filenames:
            filenames.remove('output.mp4')
        filenames = sorted(filenames, key=lambda file: int(re.findall('\d+', file)[0]))

        for i, fname in tqdm(enumerate(filenames)):

            bigger = Image.new(mode="RGB", size=(CROP_SIZE + 40, CROP_SIZE + 40), color=(0, 0, 0))
            filepath = os.path.join(images_directory, fname)
            src = Image.open(filepath)
            bigger.paste(im=src, box=(20, 20))
            bigger.save(os.path.join(tmp_savepath, fname))

            src = cv2.imread(os.path.join(tmp_savepath, fname), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_not(binary)

            contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda contour: contour.shape[0], reverse=True)

            if len(contours) >= 2 and contours[1].shape[0] > 100:
                circle_contour = contours[1]
            elif len(contours) >= 1:
                circle_contour = contours[0]
            else:
                break

            cv2.drawContours(src, [circle_contour], 0, (255, 0, 0), 3)

            x_size = gray.shape[0]
            y_size = gray.shape[1]

            M = cv2.moments(circle_contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            normalized_center_x = (center_x - 20) / (x_size - 20)
            normalized_center_y = (y_size - center_y - 20) / (y_size - 20)

            # Get PointSet
            softbody_contour = circle_contour.reshape((-1, 2))
            normalized_xs = [(pixel_index[0] - 20) / (x_size - 20) for pixel_index in softbody_contour]
            normalized_ys = [(y_size - pixel_index[1] - 20) / (y_size - 20) for pixel_index in softbody_contour]

            softbody_sparse_contour = []

            for j in range(NUM_PARTICLES):
                coordinate = compute_closest_point(normalized_center_x, normalized_center_y, normalized_xs, normalized_ys, j)
                softbody_sparse_contour.append(coordinate)

            assert np.array(softbody_sparse_contour).shape == (NUM_PARTICLES, 2)

            point_image_path = os.path.join(create_directory(os.path.join(savepath, 'points')), f'timestep_{i}_points.jpg')

            xs = [coord[0] for coord in softbody_sparse_contour]
            ys = [coord[1] for coord in softbody_sparse_contour]

            plt.figure(figsize=(3, 3), dpi=300)
            plt.axis([-0.1, 1.1, -0.1, 1.1])
            plt.scatter(xs, ys, s=0.15)
            plt.savefig(point_image_path)
            plt.close()

            cv2.imwrite(os.path.join(create_directory(os.path.join(savepath, 'contours')), f'timestep_{i}.jpg'), src)

            pointset = []
            for i in range(len(xs)):
                pointset.append([xs[i], ys[i]])
            assert len(pointset) == NUM_PARTICLES
            pointsets.append(pointset)

        ptr = write_json(pointsets, os.path.join(savepath, 'ordered_normalized_state_vectors.json'))
        ptr.close()

        remove_files = os.listdir(tmp_savepath)
        for file in remove_files:
            os.remove(os.path.join(tmp_savepath, file))
        os.rmdir(tmp_savepath)

        image_to_video([os.path.join(savepath, 'points')])
        shutil.move(src=os.path.join(savepath, 'points', 'output.mp4'), dst=os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{video_number}.mp4'))


def generate_prediction_model_data(num_input, num_output, offset):
    basepath = os.path.join(REAL_DATA_PATH, '05_postprocessed_data')
    train_video_numbers = REAL_WORLD_TRAIN_CASES
    val_video_numbers = REAL_WORLD_VAL_CASES

    # Generate Train Data
    print('===== Generating Real World Training Data ====')
    generate_data(basepath, train_video_numbers, num_input, num_output, offset, data_type='train')

    # Generate Validation Data
    print('===== Generating Real World Validation Data ====')
    generate_data(basepath, val_video_numbers, num_input, num_output, offset, data_type='val')


def generate_data(path, video_numbers, num_input, num_output, offset, data_type):

    ordered_xs = []
    ordered_ys = [[] for _ in range(num_output)]

    unordered_xs = []
    unordered_ys = [[] for _ in range(num_output)]

    sorted_xs = []
    sorted_ys = [[] for _ in range(num_output)]

    for video_number in tqdm(video_numbers):
        filepath = os.path.join(path, f'case_{video_number}', 'ordered_normalized_state_vectors.json')
        pointsets, ptr = load_json(filepath)
        ptr.close()

        num_pointset = len(pointsets)
        sequence_range = list(range(num_pointset - ((num_output + num_input - 1) * offset)))

        for timestep in sequence_range:
            x = [pointsets[timestep + offset * j] for j in range(num_input)]
            y = [pointsets[timestep + offset * (num_input + j)] for j in range(num_output)]
            ordered_xs.append(x)
            for i in range(num_output):
                ordered_ys[i].append(y[i])

            unordered_x = [shuffle_pointset(pointset) for pointset in x]
            unordered_y = [shuffle_pointset(pointset) for pointset in y]
            unordered_xs.append(unordered_x)
            for i in range(num_output):
                unordered_ys[i].append(unordered_y[i])

            sorted_x = [sort_pointset(pointset) for pointset in x]
            sorted_y = [sort_pointset(pointset) for pointset in y]
            sorted_xs.append(sorted_x)
            for i in range(num_output):
                sorted_ys[i].append(sorted_y[i])

    x_shape = np.array(ordered_xs).shape
    y_shape = np.array(ordered_ys).shape

    assert x_shape == np.array(unordered_xs).shape == np.array(sorted_xs).shape
    assert y_shape == np.array(unordered_ys).shape == np.array(sorted_ys).shape

    print(x_shape)
    print(y_shape)

    savepath = create_directory(f'../offset_{offset}_input_{num_input}_output_{num_output}')

    ptr = write_json(ordered_xs, f'{savepath}/x_{data_type}_pred_ordered.json')
    ptr.close()
    ptr = write_json(ordered_ys, f'{savepath}/y_{data_type}_pred_ordered.json')
    ptr.close()

    ptr = write_json(unordered_xs, f'{savepath}/x_{data_type}_pred_unordered.json')
    ptr.close()
    ptr = write_json(unordered_ys, f'{savepath}/y_{data_type}_pred_unordered.json')
    ptr.close()

    ptr = write_json(sorted_xs, f'{savepath}/x_{data_type}_pred_sorted.json')
    ptr.close()
    ptr = write_json(sorted_ys, f'{savepath}/y_{data_type}_pred_sorted.json')
    ptr.close()


def image_to_video(frames_paths, fps=30):
    for frames_path in frames_paths:
        filenames = os.listdir(frames_path)
        filenames = sorted(filenames, key=lambda file: int(re.findall('\d+', file)[0]))

        shape = cv2.imread(os.path.join(frames_path, filenames[0])).shape

        codec = cv2.VideoWriter_fourcc(*'MPV4')
        video = cv2.VideoWriter(os.path.join(frames_path, 'output.mp4'), codec, fps, (shape[0], shape[1]))

        for file in tqdm(filenames):
            video.write(cv2.imread(os.path.join(frames_path, file)))

        video.release()


def remove_unnecessary_frames(video_numbers, sequences):
    k = 0
    for video_number in tqdm(video_numbers):
        start_end_pair = sequences[k]

        frames_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{video_number}')
        files = os.listdir(frames_path)

        if 'output.mp4' in files:
            os.remove(os.path.join(frames_path, 'output.MP4'))

        for i in range(0, start_end_pair[0]):
            filepath = os.path.join(frames_path, f'timestep_{i}.jpg')
            if os.path.exists(filepath):
                os.remove(filepath)

        for i in range(start_end_pair[1] + 1, 899):
            filepath = os.path.join(frames_path, f'timestep_{i}.jpg')
            if os.path.exists(filepath):
                os.remove(filepath)

        k += 1
