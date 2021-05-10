from common.Constants import *
from common.Utils import load_json, normalize_pointset, create_directory
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


def draw_image_from_pointset(pointset, filepath):
    img = np.zeros((FRAME_SIZE, FRAME_SIZE))
    normalized_pointset = normalize_pointset(pointset)
    rescaled_pointset = np.array(normalized_pointset) * FRAME_SIZE
    for i, point in enumerate(rescaled_pointset):
        rescaled_pointset[i][1] = FRAME_SIZE - rescaled_pointset[i][1]

    pts = rescaled_pointset.reshape((-1, 1, 2))
    cv.fillPoly(img, pts=np.int32([pts]), color=255)
    cv.imwrite(filepath, img)


for force in FORCE_LST:
    print(f'========== Processing Force {force} =========')
    for angle in tqdm(ANGLE_LST):
        for pos in POS_LST:
            pointset_base_path = os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{pos}')
            physics, ptr = load_json(os.path.join(pointset_base_path, 'ordered_unnormalized_state_vectors.json'))
            savepath = create_directory(os.path.join(RAW_DATA_PATH, 'autoencoder_images', f'force_{force}', f'angle_{angle}', f'pos_{pos}'))

            for timestep in range(NUM_SEQUENCE_PER_ANIMATION):
                pointset = physics[timestep]
                save_filepath = os.path.join(savepath, f'timestep_{timestep}.jpg')
                draw_image_from_pointset(pointset, save_filepath)


