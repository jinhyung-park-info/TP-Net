import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from common.Utils import load_json, normalize_pointset, create_directory
from common.Constants import RAW_DATA_PATH, NUM_SEQUENCE_PER_ANIMATION
import cv2
import os

DECODER_VER = 5

model_path = os.path.join('..', 'result', 'decoder', f'version_{DECODER_VER}', 'decoder.h5')
pointsets_path = os.path.join(RAW_DATA_PATH, '..', 'simulation_data_gravity_1.6_timestep_45', 'pointset', 'force_9', 'angle_205', 'pos_265', 'ordered_unnormalized_state_vectors.json')
pointsets, ptr = load_json(pointsets_path)
ptr.close()

decoder = load_model(model_path, compile=False)
savepath = create_directory('./quick_decoder_result')
for i in range(NUM_SEQUENCE_PER_ANIMATION):

    normalized_pointset = normalize_pointset(pointsets[i])
    result = decoder.predict(np.array([normalized_pointset]))[0]
    result = result * 255.0
    cv2.imwrite(os.path.join(savepath, f'reconstructed_{i}.jpg'), result)
