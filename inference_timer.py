from common.Utils import create_directory
from tensorflow.keras.models import load_model
import argparse
import os
from real_world_test_utils import get_real_world_input_pointset, update_input_pointset
from time import time, strftime, localtime

parser = argparse.ArgumentParser(description='Inference Time Calculator')
parser.add_argument('--model_type', required=False, default='global_pointnet', choices=['global_pointnet', 'lstm'])
parser.add_argument('--model_ver', required=False, default=13)
parser.add_argument('--fine_tuning_ver', required=False, default=3)
parser.add_argument('--data_type', required=False, default='unordered')
parser.add_argument('--video_number', required=False, default=15)
parser.add_argument('--num_iterations', required=False, default=1000)
FLAGS = parser.parse_args()

MODEL_TYPE = FLAGS.model_type
MODEL_VER = int(FLAGS.model_ver)
FINE_TUNE_VER = int(FLAGS.fine_tuning_ver)
DATA_TYPE = FLAGS.data_type
VIDEO_NUMBER = int(FLAGS.video_number)
NUM_ITERATIONS = int(FLAGS.num_iterations)

model_path = os.path.join('result', f'{MODEL_TYPE}', f'version_{MODEL_VER}', 'fine_tuning', f'version_{FINE_TUNE_VER}', f'{MODEL_TYPE}_model_real.h5')
model = load_model(model_path, compile=False)

input_pointset = get_real_world_input_pointset(case=VIDEO_NUMBER, num_input_frames=3, offset=2, test_data_type=DATA_TYPE)

# Warm Up
for i in range(10):
    predicted_pointset = model.predict(input_pointset)[0]
    input_pointset = update_input_pointset(input_pointset, predicted_pointset)

# Inference Time
print('========= Measuring Inference Time =========')
start = time()
for i in range(NUM_ITERATIONS):
    predicted_pointset = model.predict(input_pointset)[0]
    input_pointset = update_input_pointset(input_pointset, predicted_pointset)
end = time()

speed_per_frame = (end - start) / NUM_ITERATIONS
fps = 1 / speed_per_frame

print(f'{NUM_ITERATIONS} Inference Time : {end - start}')
print(f'Average Inference Time Per Frame: {speed_per_frame}')
print(f'FPS: {fps}', end='\n\n')

# Record Result
print('========= Recording Results to File =========')
savepath = create_directory(os.path.join('result', 'inference_time'))
file = open(os.path.join(savepath, f'{MODEL_TYPE}_{MODEL_VER}_fine_tuning_{FINE_TUNE_VER}_{DATA_TYPE}.txt'), 'w')

file.write('======== Inference Time ========\n\n')
file.write('Date: {}\n'.format(strftime('%Y-%m-%d', localtime(time()))))
file.write(f'Model Type          : {MODEL_TYPE}\n')
file.write(f'Data Type           : {DATA_TYPE}\n')
file.write(f'Model Ver           : {MODEL_VER}\n')
file.write(f'Fine Tuning Ver     : {FINE_TUNE_VER}\n\n')

file.write(f'Tested Real World Video Number : {VIDEO_NUMBER}\n')
file.write(f'Inferred Iterations : {NUM_ITERATIONS}\n\n')

file.write(f'{NUM_ITERATIONS} Inference Time : {end - start}\n')
file.write(f'Average Inference Time Per Frame: {speed_per_frame}\n\n')

file.write(f'FPS: {fps}\n')
file.close()
