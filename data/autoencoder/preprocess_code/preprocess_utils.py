from common.Constants import *
import random
import numpy as np
from common.Utils import normalize_pointset, load_json, sort_pointset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def shuffle_pointset(pointset):
    return random.sample(pointset, len(pointset))


def generate_autoencoder_data(num_use_animations, case_per_animation, mode):

    print('============== Generating Autoencoder Data ===============')

    img_filepaths = []
    labels = []

    total_cases = []
    for force in FORCE_LST:
        for angle in ANGLE_LST:
            for pos in POS_LST:
                total_cases.append((force, angle, pos))

    total_cases.remove((9, 205, 265))   # test case

    selected_animations = random.sample(total_cases, k=num_use_animations)

    for selected_animation in tqdm(selected_animations):

        cases = random.sample(list(range(0, NUM_SEQUENCE_PER_ANIMATION)), k=case_per_animation)

        pointset_path = os.path.join(RAW_DATA_PATH,
                                     'pointset',
                                     f'force_{selected_animation[0]}',
                                     f'angle_{selected_animation[1]}',
                                     f'pos_{selected_animation[2]}',
                                     'ordered_unnormalized_state_vectors.json')
        pointset, ptr = load_json(pointset_path)
        ptr.close()

        for timestep in cases:
            image_path = os.path.join(RAW_DATA_PATH,
                                      'autoencoder_images',
                                      f'force_{selected_animation[0]}',
                                      f'angle_{selected_animation[1]}',
                                      f'pos_{selected_animation[2]}',
                                      f'timestep_{timestep}.jpg')

            img_filepaths.append(image_path)

            if mode == 'unordered':
                labels.append(shuffle_pointset(normalize_pointset(pointset[timestep])))
            elif mode == 'sorted':
                labels.append(sort_pointset(normalize_pointset(pointset[timestep])))
            else:
                labels.append(normalize_pointset(pointset[timestep]))

    shuffled_img_filepaths, shuffled_labels = shuffle(img_filepaths, labels)
    shuffled_img_filepaths = np.array(shuffled_img_filepaths)
    shuffled_labels = np.array(shuffled_labels)

    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        shuffled_img_filepaths,
        shuffled_labels,
        test_size=0.2,
        random_state=RANDOM_SEED)

    print("=============== Writing To Files =============")

    np.save(f'../{mode}/train_pointset_{mode}.npy', train_labels)
    np.save(f'../{mode}/val_pointset_{mode}.npy', val_labels)
    np.save(f'../{mode}/train_img_paths.npy', train_img_paths)
    np.save(f'../{mode}/val_img_paths.npy', val_img_paths)

    print(train_img_paths.shape)
    print(val_img_paths.shape)

    print(train_labels.shape)
    print(val_labels.shape)

    print("=============== Done =============")
