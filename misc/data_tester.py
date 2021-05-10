from common.Constants import *
from common.Utils import load_json
import os
import numpy as np
from tqdm import tqdm

for force in tqdm(FORCE_LST):
    for angle in ANGLE_LST:
        for pos in POS_LST:
            physics, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{pos}', 'ordered_unnormalized_state_vectors.json'))
            ptr.close()
            shape = np.array(physics).shape
            assert shape == (NUM_SEQUENCE_PER_ANIMATION, NUM_PARTICLES * 2), f'Received {shape} in Case {force, angle, pos}'

            num_images = len(os.listdir(os.path.join(RAW_DATA_PATH, 'grey_images', f'force_{force}', f'angle_{angle}', f'pos_{pos}')))
            assert num_images == NUM_SEQUENCE_PER_ANIMATION, f'Recevied {num_images} images in Case {force, angle, pos}'
