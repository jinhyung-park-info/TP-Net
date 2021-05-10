import cv2 as cv
from common.Utils import create_directory

ALGORITHM = 'KNN'
VIDEO_NAME = 'sample_2'

VIDEO_PATH = f'D:real_world/raw_video/{VIDEO_NAME}/{VIDEO_NAME}.MP4'
OUT_PATH = create_directory(f'D:real_world/background_subtracted/{VIDEO_NAME}')

if ALGORITHM == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

raw_video = cv.VideoCapture(cv.samples.findFileOrKeep(VIDEO_PATH))

width = raw_video.get(cv.CAP_PROP_FRAME_WIDTH)
height = raw_video.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = raw_video.get(cv.CAP_PROP_FPS)
codec = cv.VideoWriter_fourcc(*'MPV4')

background_subtracted_video = cv.VideoWriter(f'{OUT_PATH}/{VIDEO_NAME}_{ALGORITHM}.MP4', codec, fps, (int(width), int(height)))

if not raw_video.isOpened():
    print('Unable to open: ' + VIDEO_PATH)
    exit(0)

count = 0

while True:
    ret, frame = raw_video.read()
    if frame is None or count >= 4000:
        break

    if 1000 < count < 4000:
        fgMask = backSub.apply(frame)

        # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv.putText(frame, str(raw_video.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # cv.namedWindow('Resized Frame', cv.WINDOW_NORMAL)
        # cv.namedWindow('Resized FG Mask', cv.WINDOW_NORMAL)

        # cv.resizeWindow('Frame', 2000, 800)
        # cv.resizeWindow('FG Mask', 2000, 800)

        # cv.imshow('Resized Frame', frame)
        # cv.imshow('Resized FG Mask', fgMask)
        background_subtracted_video.write(fgMask)
    count += 1

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

    if count % 1000 == 0:
        print(f'Passed {count}')

background_subtracted_video.release()
cv.destroyAllWindows()
