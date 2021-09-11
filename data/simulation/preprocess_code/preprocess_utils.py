from common.Constants import *
from common.Utils import load_json, is_within_collision_range, normalize_pointset, shuffle_pointset, create_directory
import numpy as np
from tqdm import tqdm
import random


def generate_prediction_model_data(num_animations, n_input, n_output, offset):
    print('========== Identifying Collision Info from Dataset ===========')
    collision_dict = identify_collision_timestep(n_input, n_output, offset)

    print('========== Splitting Animation Dataset to Train, Val, Test Cases ===========')
    partial_train_cases, full_train_cases, val_cases, test_cases = split_animations(num_animations)

    print('==================== Generating Train Data =========================')
    generate_train_data(collision_dict, partial_train_cases, full_train_cases, n_input, n_output, offset)

    print('==================== Generating Val Data =========================')
    generate_val_data(val_cases, n_input, n_output, offset)

    print('=================== Done ======================')


def identify_collision_timestep(n_input, n_output, offset):

    is_near_collision = {}

    for force in FORCE_LST:
        for angle in ANGLE_LST:
            for x_pos, y_pos in POS_LST:
                pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{x_pos}_{y_pos}', 'ordered_unnormalized_pointset.json'))
                ptr.close()

                interval = []

                for i in range(NUM_SEQUENCE_PER_ANIMATION - ((n_output + n_input - 1) * offset)):
                    if is_within_collision_range(pointsets[i]):
                        interval.append(i)

                is_near_collision[(force, angle, x_pos, y_pos)] = interval

    return is_near_collision


def split_animations(num_animations):

    num_val_animations = 150
    num_test_animations = 100
    num_train_animations = 6875
    assert num_animations >= num_train_animations + num_test_animations + num_val_animations

    num_full_train_animations = int(0.035 * num_train_animations)
    num_partial_train_animations = num_train_animations - num_full_train_animations

    total_cases = []
    for force in FORCE_LST:
        for angle in ANGLE_LST:
            for x_pos, y_pos in POS_LST:
                total_cases.append((force, angle, x_pos, y_pos))

    assert len(total_cases) == num_animations

    val_cases = random.sample(total_cases, k=num_val_animations)
    for pair in val_cases:
        total_cases.remove(pair)

    test_cases = random.sample(total_cases, k=num_test_animations)
    for pair in test_cases:
        total_cases.remove(pair)

    train_cases = random.sample(total_cases, k=num_train_animations)

    assert len(test_cases) == num_test_animations
    assert len(val_cases) == num_val_animations
    assert len(train_cases) == num_train_animations

    full_train_cases = random.sample(train_cases, k=num_full_train_animations)
    for case in full_train_cases:
        train_cases.remove(case)
    partial_train_cases = train_cases

    assert len(partial_train_cases) == num_partial_train_animations
    assert len(full_train_cases) == num_full_train_animations

    return partial_train_cases, full_train_cases, val_cases, test_cases


def generate_train_data(collision_dict,
                        partial_train_cases,
                        full_train_cases,
                        n_input,
                        n_output,
                        offset,
                        normal_case_per_animation=5,
                        collision_case_per_animation=15):

    sequence_range = list(range(NUM_SEQUENCE_PER_ANIMATION - ((n_output + n_input - 1) * offset)))

    x_train_unordered = []
    y_train_unordered = [[] for _ in range(n_output)]

    x_train_ordered = []
    y_train_ordered = [[] for _ in range(n_output)]

    # Generating Partially Trained Animation Data
    for train_case in tqdm(partial_train_cases):

        pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH,
                                                'pointset',
                                                f'force_{train_case[0]}',
                                                f'angle_{train_case[1]}',
                                                f'pos_{train_case[2]}_{train_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        # Sample timesteps from the entire animation interval
        normal_timesteps = random.sample(sequence_range, k=normal_case_per_animation)

        for timestep in normal_timesteps:
            x = [normalize_pointset(pointsets[timestep + offset * j]) for j in range(n_input)]
            y = [normalize_pointset(pointsets[timestep + offset * (n_input + j)]) for j in range(n_output)]
            x_train_ordered.append(x)
            for i in range(n_output):
                y_train_ordered[i].append(y[i])

            # unordered
            unordered_x = [shuffle_pointset(pointset) for pointset in x]
            unordered_y = [shuffle_pointset(pointset) for pointset in y]
            x_train_unordered.append(unordered_x)
            for i in range(n_output):
                y_train_unordered[i].append(unordered_y[i])

        collision_timesteps = random.sample(collision_dict[train_case], k=min(collision_case_per_animation, len(collision_dict[train_case])))

        # remove same timesteps from sampled normal_timesteps
        for timestep in collision_timesteps:
            if timestep in normal_timesteps:
                collision_timesteps.remove(timestep)

        for timestep in collision_timesteps:
            x = [normalize_pointset(pointsets[timestep + offset * j]) for j in range(n_input)]
            y = [normalize_pointset(pointsets[timestep + offset * (n_input + j)]) for j in range(n_output)]
            x_train_ordered.append(x)
            for i in range(n_output):
                y_train_ordered[i].append(y[i])

            # unordered
            unordered_x = [shuffle_pointset(pointset) for pointset in x]
            unordered_y = [shuffle_pointset(pointset) for pointset in y]
            x_train_unordered.append(unordered_x)
            for i in range(n_output):
                y_train_unordered[i].append(unordered_y[i])

    # Generating Fully Trained Animation Data
    for train_case in tqdm(full_train_cases):

        pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH,
                                                'pointset',
                                                f'force_{train_case[0]}',
                                                f'angle_{train_case[1]}',
                                                f'pos_{train_case[2]}_{train_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        for timestep in sequence_range:
            x = [normalize_pointset(pointsets[timestep + offset * j]) for j in range(n_input)]
            y = [normalize_pointset(pointsets[timestep + offset * (n_input + j)]) for j in range(n_output)]
            x_train_ordered.append(x)
            for i in range(n_output):
                y_train_ordered[i].append(y[i])

            # unordered
            unordered_x = [shuffle_pointset(pointset) for pointset in x]
            unordered_y = [shuffle_pointset(pointset) for pointset in y]
            x_train_unordered.append(unordered_x)
            for i in range(n_output):
                y_train_unordered[i].append(unordered_y[i])

    print('=========== Converting it to Numpy Arrays ===========')

    x_train_unordered = np.array(x_train_unordered)
    y_train_unordered = np.array(y_train_unordered)

    x_train_ordered = np.array(x_train_ordered)
    y_train_ordered = np.array(y_train_ordered)

    assert x_train_unordered.shape == x_train_ordered.shape
    assert y_train_unordered.shape == y_train_ordered.shape

    print(x_train_unordered.shape)
    print(y_train_unordered.shape)

    if n_output == 1:
        y_train_ordered = y_train_ordered[0]
        y_train_unordered = y_train_unordered[0]

    savepath = create_directory(f'../preprocessed_data/input_{n_input}/offset_{offset}_num_pred_{n_output}')

    print('=========== Saving Unordered Train Data ===========')
    np.save(f'{savepath}/x_train_pred_unordered.npy', x_train_unordered)
    np.save(f'{savepath}/y_train_pred_unordered.npy', y_train_unordered)

    print('=========== Saving Ordered Train Data ===========')
    np.save(f'{savepath}/x_train_pred_ordered.npy', x_train_ordered)
    np.save(f'{savepath}/y_train_pred_ordered.npy', y_train_ordered)

    return str(x_train_unordered.shape), str(y_train_unordered.shape)


def generate_val_data(val_cases, n_input, n_output, offset):

    sequence_range = list(range(NUM_SEQUENCE_PER_ANIMATION - ((n_output + n_input - 1) * offset)))

    x_val_unordered = []
    y_val_unordered = [[] for i in range(n_output)]

    x_val_ordered = []
    y_val_ordered = [[] for i in range(n_output)]

    for val_case in tqdm(val_cases):

        pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH,
                                                'pointset',
                                                f'force_{val_case[0]}',
                                                f'angle_{val_case[1]}',
                                                f'pos_{val_case[2]}_{val_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        for timestep in sequence_range:
            x = [normalize_pointset(pointsets[timestep + offset * j]) for j in range(n_input)]
            y = [normalize_pointset(pointsets[timestep + offset * (n_input + j)]) for j in range(n_output)]
            x_val_ordered.append(x)
            for i in range(n_output):
                y_val_ordered[i].append(y[i])

            # unordered
            unordered_x = [shuffle_pointset(pointset) for pointset in x]
            unordered_y = [shuffle_pointset(pointset) for pointset in y]
            x_val_unordered.append(unordered_x)
            for i in range(n_output):
                y_val_unordered[i].append(unordered_y[i])

    print('=========== Converting it to Numpy Arrays ===========')

    x_val_unordered = np.array(x_val_unordered)
    y_val_unordered = np.array(y_val_unordered)

    x_val_ordered = np.array(x_val_ordered)
    y_val_ordered = np.array(y_val_ordered)

    assert x_val_unordered.shape == x_val_ordered.shape
    assert y_val_unordered.shape == y_val_ordered.shape

    print(x_val_unordered.shape)
    print(y_val_unordered.shape)

    if n_output == 1:
        y_val_ordered = y_val_ordered[0]
        y_val_unordered = y_val_unordered[0]

    savepath = create_directory(f'../preprocessed_data/input_{n_input}/offset_{offset}_num_pred_{n_output}')

    print('=========== Saving Unordered Val Data ===========')
    np.save(f'{savepath}/x_val_pred_unordered.npy', x_val_unordered)
    np.save(f'{savepath}/y_val_pred_unordered.npy', y_val_unordered)

    print('=========== Saving Ordered Val Data ===========')
    np.save(f'{savepath}/x_val_pred_ordered.npy', x_val_ordered)
    np.save(f'{savepath}/y_val_pred_ordered.npy', y_val_ordered)

    return str(x_val_unordered.shape), str(y_val_unordered.shape)


def find_min_max():

    total_pointsets = []

    for force in FORCE_LST:
        for angle in tqdm(ANGLE_LST):
            for x_pos, y_pos in POS_LST:
                pointsets, ptr = load_json(f'{SIM_DATA_PATH}/pointset/force_{force}/angle_{angle}/pos_{x_pos}_{y_pos}/ordered_unnormalized_pointset.json')
                ptr.close()
                total_pointsets.extend(pointsets)

    total_pointsets = np.array(total_pointsets)
    return np.max(total_pointsets), np.min(total_pointsets)
