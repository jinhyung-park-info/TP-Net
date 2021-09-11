from common.Utils import load_json
from tensorflow.keras.models import load_model
from time import time
import numpy as np
import os
import cv2
from common.Constants import CODEC

STEPS = 1000
WARM_UP_STEPS = 80
NUM_INPUT = 4
NUM_PARTICLES = 30
OFFSET = 4
CROP_SIZE = 922
DISTANCE = 1624
HEIGHT = 941
PATH_TO_VIDEO_DATA = os.path.join('./sample_video')
PATH_TO_SAVED_MODEL = os.path.join(f'./result/global_pointnet/global_pointnet-{NUM_INPUT}/seed_2/global_pointnet_model.h5')
RESULT_SAVEPATH = './sample_video'


def convert_to_pixel_coordinates(predicted_pointset, height):
    pixel_pointset = []

    for point in predicted_pointset:
        x = int(CROP_SIZE * point[0])
        y = int(CROP_SIZE * (1 - point[1]))
        pixel_pointset.append([x, y])

    for i in range(NUM_PARTICLES):
        pixel_pointset[i][0] += 228
        pixel_pointset[i][1] += height - CROP_SIZE

    return pixel_pointset


if __name__ == '__main__':

    # 1. Load Model
    model = load_model(PATH_TO_SAVED_MODEL, compile=False)

    # 2. Load a test case
    input_point_sets, ptr = load_json(os.path.join(PATH_TO_VIDEO_DATA, 'ordered_normalized_state_vectors.json'))
    ptr.close()
    input_point_sets = np.array([[input_point_sets[OFFSET * i] for i in range(NUM_INPUT)]])

    # 3. Load Background image to which the predicted points will be plotted
    background_images = [cv2.imread(os.path.join(PATH_TO_VIDEO_DATA, 'background_image.jpg'), cv2.COLOR_BGR2RGB) for _ in range(STEPS + WARM_UP_STEPS + NUM_INPUT)]

    # 4. Warm Up for Measuring FPS
    print('========= Warming Up =========')
    output_video = cv2.VideoWriter(os.path.join(RESULT_SAVEPATH, f'predicted_video_input_{NUM_INPUT}.MP4'), CODEC, 30, (1150, 1040))
    for i in range(NUM_INPUT):
        pixel_coordinates = convert_to_pixel_coordinates(input_point_sets[0][i], HEIGHT)
        for k in range(NUM_PARTICLES):
            background_images[i] = cv2.line(background_images[i], pixel_coordinates[k], pixel_coordinates[k], (47, 164, 193), thickness=12)
        output_video.write(background_images[i])

    for i in range(0, WARM_UP_STEPS, 8):
        predicted_pointset = model.predict(input_point_sets)

        for j in range(8):
            pixel_coordinates = convert_to_pixel_coordinates(predicted_pointset[j][0], HEIGHT)
            for k in range(NUM_PARTICLES):
                background_images[NUM_INPUT + i + j] = cv2.line(background_images[NUM_INPUT + i + j], pixel_coordinates[k], pixel_coordinates[k], (47, 164, 193), thickness=12)
            output_video.write(background_images[NUM_INPUT + i + j])

        input_point_sets = np.array(predicted_pointset[-(NUM_INPUT):]).reshape((1, NUM_INPUT, NUM_PARTICLES, 2))

    # 5. Measure FPS
    print('========= Measuring FPS =========')
    count = 0
    start = time()
    for i in range(0, STEPS, 8):
        predicted_pointset = model.predict(input_point_sets)

        for j in range(8):
            pixel_coordinates = convert_to_pixel_coordinates(predicted_pointset[j][0], HEIGHT)
            for k in range(NUM_PARTICLES):
                background_images[NUM_INPUT + WARM_UP_STEPS + i + j] = cv2.line(background_images[NUM_INPUT + WARM_UP_STEPS + i + j], pixel_coordinates[k], pixel_coordinates[k], (47, 164, 193), thickness=12)
            output_video.write(background_images[NUM_INPUT + WARM_UP_STEPS + i + j])

        input_pointset = np.array(predicted_pointset[-(NUM_INPUT):]).reshape((1, NUM_INPUT, NUM_PARTICLES, 2))

    end = time()

    # 6. Save Video
    output_video.release()
    cv2.destroyAllWindows()

    total_time_fps = end - start
    speed_per_frame = total_time_fps / STEPS
    fps = 1 / speed_per_frame

    print(f'Total Time : {end - start}')
    print(f'Average Time Per Frame: {speed_per_frame}')
    print(f'FPS: {fps}', end='\n\n')

    # 7. Warming up for measuring IPS
    print('========= Warming Up =========')
    for i in range(0, 200, 8):
        predicted_pointset = model.predict(input_point_sets)
        input_pointset = np.array(predicted_pointset[-(NUM_INPUT):]).reshape((1, NUM_INPUT, NUM_PARTICLES, 2))

    # 8. Measure IPS
    print('========= Measuring IPS =========')
    count = 0
    start = time()
    for i in range(0, STEPS, 8):
        predicted_pointset = model.predict(input_point_sets)
        input_pointset = np.array(predicted_pointset[-(NUM_INPUT):]).reshape((1, NUM_INPUT, NUM_PARTICLES, 2))

    end = time()
    total_time_ips = end - start
    speed_per_inference = total_time_ips / STEPS
    ips = 1 / speed_per_inference

    print(f'Total Time : {total_time_ips}')
    print(f'Average Time Per Inference: {speed_per_inference}')
    print(f'IPS: {ips}', end='\n\n')

    # 9. Record Result
    print('========= Recording Results to File =========')
    file = open(os.path.join(RESULT_SAVEPATH, f'inference_time_input_{NUM_INPUT}.txt'), 'w')

    file.write('======== Inferences Per Second (IPS) ========\n\n')
    file.write(f'Inferred Steps             : {STEPS}\n')
    file.write(f'Total Elapsed Time         : {total_time_ips}\n')
    file.write(f'Average Time Per Inference : {speed_per_inference}\n')
    file.write(f'          IPS              : {ips}\n\n\n')

    file.write('======== Frames Per Second (FPS) ========\n\n')
    file.write(f'Inferred Frames            : {STEPS}\n')
    file.write(f'Total Elapsed Time         : {total_time_fps}\n')
    file.write(f'Average Time Per Frame     : {speed_per_frame}\n')
    file.write(f'          FPS              : {fps}\n\n\n')
    file.close()
