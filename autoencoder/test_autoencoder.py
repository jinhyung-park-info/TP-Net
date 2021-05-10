from common.Constants import *
from common.Utils import load_json, normalize_pointset, create_directory
from PIL import Image
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ENCODER_VER = 1
TEST_CASES = [(9, 205, 265)] + TEST_CASES_FOR_UNORDERED


def plot_points(gt_points, predicted_points, savepath, timestep):
    gt_xs = [coordinate[0] for coordinate in gt_points]
    gt_ys = [coordinate[1] for coordinate in gt_points]

    pd_xs = [coordinate[0] for coordinate in predicted_points]
    pd_ys = [coordinate[1] for coordinate in predicted_points]

    plt.axis([0, 1.0, 0, 1])
    plt.scatter(pd_xs, pd_ys, s=1.0, label='prediction')
    plt.scatter(gt_xs, gt_ys, s=1.0, label='ground_truth')
    plt.legend()

    savepath = create_directory(os.path.join(savepath, 'points_comparison'))
    filepath = os.path.join(savepath, f'timestep_{timestep}.jpg')
    plt.savefig(filepath)
    plt.clf()

    return Image.open(filepath).resize((VIDEO_HEIGHT, VIDEO_HEIGHT))


def plot_prediction_points(predicted_points, savepath, timestep):

    pd_xs = [coordinate[0] for coordinate in predicted_points]
    pd_ys = [coordinate[1] for coordinate in predicted_points]

    plt.axis([0, 1.0, 0, 1])
    plt.scatter(pd_xs, pd_ys, s=1.0, label='prediction')
    plt.legend()

    savepath = create_directory(os.path.join(savepath, 'points_comparison'))
    filepath = os.path.join(savepath, f'timestep_{timestep}_prediction.jpg')
    plt.savefig(filepath)
    plt.clf()

    return Image.open(filepath).resize((VIDEO_HEIGHT, VIDEO_HEIGHT))


base_path = os.path.join('../result', 'encoder', f'version_{ENCODER_VER}')
encoder = load_model(os.path.join(base_path, 'encoder.h5'), compile=False)
plt.figure(figsize=(3, 3), dpi=300)

for i, case in enumerate(TEST_CASES):
    save_path = create_directory(os.path.join(base_path, 'prediction_result', f'Test_Case_{i}'))
    physics, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{case[0]}', f'angle_{case[1]}', f'pos_{case[2]}', 'ordered_unnormalized_state_vectors.json'))
    ptr.close()
    input_image_path = os.path.join(RAW_DATA_PATH, 'autoencoder_images', f'force_{case[0]}', f'angle_{case[1]}', f'pos_{case[2]}')
    output_video = cv2.VideoWriter(os.path.join(save_path, '../..', f'Test Case_{i}.MP4'), CODEC, 30, (VIDEO_HEIGHT * 3, VIDEO_HEIGHT))

    for timestep in tqdm(range(NUM_SEQUENCE_PER_ANIMATION)):
        merged = Image.new(mode="RGB", size=(VIDEO_HEIGHT * 3, VIDEO_HEIGHT), color=(255, 255, 255))
        input_image = Image.open(os.path.join(input_image_path, f'timestep_{timestep}.jpg')).resize((FRAME_SIZE, FRAME_SIZE))
        merged.paste(im=input_image, box=(WALL_SIZE, WALL_SIZE))

        # Ground Truth
        ground_truth_points = normalize_pointset(physics[timestep])

        # Prediction
        normalized_image = cv2.imread(os.path.join(input_image_path, f'timestep_{timestep}.jpg'), 0) / 255.0
        predicted_pointset = encoder.predict(np.array([normalized_image]))[0]
        assert predicted_pointset.shape == (20, 2)

        plotted_image = plot_points(ground_truth_points, predicted_pointset, save_path, timestep)
        merged.paste(im=plotted_image, box=(VIDEO_HEIGHT, 0))

        prediction_image = plot_prediction_points(predicted_pointset, save_path, timestep)
        merged.paste(im=prediction_image, box=(VIDEO_HEIGHT * 2, 0))
        merged_savepath = create_directory(os.path.join(save_path, 'merged'))
        merged.save(os.path.join(merged_savepath, f'timestep_{timestep}.jpg'))
        output_video.write(np.array(merged))

    output_video.release()
    cv2.destroyAllWindows()









