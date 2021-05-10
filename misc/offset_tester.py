import cv2
from tqdm import tqdm
import os

PATHS = './pos_69/'
OFFSET = 4


def image_to_video(frames_path, offset, fps=30):
    for filename in os.listdir(frames_path):

        print(filename)

        shape = cv2.imread(os.path.join(frames_path, filename)).shape

        codec = cv2.VideoWriter_fourcc(*'MPV4')
        video = cv2.VideoWriter(os.path.join(frames_path, 'output.mp4'), codec, fps, (shape[0], shape[1]))

        for i in range(0, 200, offset):
            video.write(cv2.imread(os.path.join(frames_path, f'timestep_{i}.jpg')))

        video.release()
        print('finished')
        break


image_to_video(PATHS, OFFSET)
