import numpy as np
import cv2
import os
from simulation_test_utils import draw_box2d_image
from PIL import Image
from common.Constants import WALL_SIZE

CODEC = cv2.VideoWriter_fourcc(*'MPV4')
FPS = 30

LOC_MAX = 43.8562
LOC_MIN = 1.1438


def unnormalize(data, data_max, data_min):
    return (data + 1) * (data_max - data_min) / 2 + data_min

predicted_values = np.load('./softbody_predictions.npy')

for test_case in range(20):
    output_video = cv2.VideoWriter(os.path.join(f'Test Case_{test_case + 1}.MP4'), CODEC, FPS, (900, 900))
    for timestep in range(100):
        predicted_loc_vel = predicted_values[test_case][0][timestep]
        predicted_pointset = predicted_loc_vel[:, :2]
        unnormalized_predicted_pointset = unnormalize(predicted_pointset, LOC_MAX, LOC_MIN)

        predicted_frame = draw_box2d_image(unnormalized_predicted_pointset)
        image = Image.new(mode="L", size=(900, 900), color=255)
        image.paste(im=Image.fromarray(predicted_frame), box=(WALL_SIZE, WALL_SIZE))
        output_video.write(np.array(image))
