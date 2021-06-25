from common.Constants import *
from common.Utils import load_json, is_within_collision_range, write_json, create_directory, get_nested_pointset
import numpy as np
from tqdm import tqdm
import random

NUM_INPUT_FRAMES = 9
NUM_EVAL_FRAMES = 40 + 150


def get_velocity_from_position(position_vector, include_velocity):
    assert len(position_vector) == NUM_INPUT_FRAMES + 1
    new_vector = []

    for i in range(NUM_INPUT_FRAMES):
        new_vector.append([])
        for j, coordinate in enumerate(position_vector[i]):
            if include_velocity:
                assert coordinate[0] == position_vector[i][j][0]
                assert coordinate[1] == position_vector[i][j][1]

                vel_x = position_vector[i+1][j][0] - position_vector[i][j][0]
                vel_y = position_vector[i+1][j][1] - position_vector[i][j][1]
                velocity_vector = [vel_x, vel_y]
            else:
                velocity_vector = [0, 0]
            new_vector[i].append(velocity_vector)

    assert np.array(new_vector).shape == (NUM_INPUT_FRAMES, NUM_PARTICLES, 2)

    return new_vector


def get_velocity_from_position_for_eval(position_vector, include_velocity):
    assert len(position_vector) == NUM_EVAL_FRAMES + 1
    new_vector = []

    for i in range(NUM_EVAL_FRAMES):
        new_vector.append([])
        for j, coordinate in enumerate(position_vector[i]):
            if include_velocity:
                assert coordinate[0] == position_vector[i][j][0]
                assert coordinate[1] == position_vector[i][j][1]

                vel_x = position_vector[i+1][j][0] - position_vector[i][j][0]
                vel_y = position_vector[i+1][j][1] - position_vector[i][j][1]
                velocity_vector = [vel_x, vel_y]
            else:
                velocity_vector = [0, 0]
            new_vector[i].append(velocity_vector)

    assert np.array(new_vector).shape == (NUM_EVAL_FRAMES, NUM_PARTICLES, 2)

    return new_vector


def generate_prediction_model_data_for_dnri(num_animations, offset, include_vel=True, only_eval_data=False):

    if not only_eval_data:
        print('========== Identifying Collision Info from Dataset ===========')
        collision_dict = identify_collision_timestep(offset)

        print('========== Splitting Animation Dataset to Train, Val, Test Cases ===========')
        partial_train_cases, full_train_cases, val_cases, test_cases = split_animations(num_animations)

        print('==================== Generating Train Data =========================')
        generate_train_data(collision_dict, partial_train_cases, full_train_cases, offset, include_vel)

        print('==================== Generating Val Data =========================')
        generate_val_data(val_cases, offset, include_vel)

        print('==================== Generating Test Data =========================')
        generate_test_data(test_cases, offset, include_vel)

    else:
        generate_eval_data(offset, include_vel)

    print('=================== Done ======================')


def identify_collision_timestep(offset):

    is_near_collision = {}

    for force in FORCE_LST:
        for angle in ANGLE_LST:
            for x_pos, y_pos in POS_LST:
                pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{x_pos}_{y_pos}', 'ordered_unnormalized_pointset.json'))
                ptr.close()

                interval = []

                for i in range(NUM_SEQUENCE_PER_ANIMATION - ((NUM_INPUT_FRAMES + 1) * offset)):
                    if is_within_collision_range(pointsets[i]):
                        interval.append(i)

                is_near_collision[(force, angle, x_pos, y_pos)] = interval

    return is_near_collision


def split_animations(num_animations):

    num_val_animations = 150
    num_test_animations = 100
    num_train_animations = 6875  # 6875
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
                        offset,
                        include_vel,
                        normal_case_per_animation=5,
                        collision_case_per_animation=15):

    sequence_range = list(range(NUM_SEQUENCE_PER_ANIMATION - ((NUM_INPUT_FRAMES + 1) * offset)))

    locations = []
    velocities = []

    # Generating Partially Trained Animation Data
    for train_case in tqdm(partial_train_cases):

        pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH,
                                                'pointset',
                                                f'force_{train_case[0]}',
                                                f'angle_{train_case[1]}',
                                                f'pos_{train_case[2]}_{train_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        # Sample timesteps from the entire animation interval
        normal_timesteps = random.sample(sequence_range, k=normal_case_per_animation)

        for timestep in normal_timesteps:
            position_vector = [get_nested_pointset(pointsets[timestep + offset * j]) for j in range(NUM_INPUT_FRAMES + 1)]
            velocity_vector = get_velocity_from_position(position_vector, include_vel)
            locations.append(position_vector[:-1])
            velocities.append(velocity_vector)

        collision_timesteps = random.sample(collision_dict[train_case], k=min(collision_case_per_animation, len(collision_dict[train_case])))

        # remove same timesteps from sampled normal_timesteps
        for timestep in collision_timesteps:
            if timestep in normal_timesteps:
                collision_timesteps.remove(timestep)

        for timestep in collision_timesteps:
            position_vector = [get_nested_pointset(pointsets[timestep + offset * j]) for j in range(NUM_INPUT_FRAMES + 1)]
            velocity_vector = get_velocity_from_position(position_vector, include_vel)
            locations.append(position_vector[:-1])
            velocities.append(velocity_vector)

    # Generating Fully Trained Animation Data
    for train_case in tqdm(full_train_cases):

        pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH,
                                                'pointset',
                                                f'force_{train_case[0]}',
                                                f'angle_{train_case[1]}',
                                                f'pos_{train_case[2]}_{train_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        for timestep in sequence_range:
            position_vector = [get_nested_pointset(pointsets[timestep + offset * j]) for j in range(NUM_INPUT_FRAMES + 1)]
            velocity_vector = get_velocity_from_position(position_vector, include_vel)
            locations.append(position_vector[:-1])
            velocities.append(velocity_vector)

    print('=========== Checking Train Data Type ===========')
    train_loc_shape = np.array(locations).shape
    train_vel_shape = np.array(velocities).shape

    print(train_loc_shape)
    print(train_vel_shape)

    savepath = create_directory(f'../preprocessed_data_dnri/offset_{offset}_input_{NUM_INPUT_FRAMES}')

    print('=========== Saving dNRI Train Data ===========')
    train_loc = np.array(locations)
    train_vel = np.array(velocities)

    if include_vel:
        np.save(os.path.join(savepath, 'loc_train_mine'), train_loc)
        np.save(os.path.join(savepath, 'vel_train_mine'), train_vel)
    else:
        np.save(os.path.join(savepath, 'loc_train_mine_zeros'), train_loc)
        np.save(os.path.join(savepath, 'vel_train_mine_zeros'), train_vel)

    return


def generate_val_data(val_cases, offset, include_vel):

    sequence_range = list(range(NUM_SEQUENCE_PER_ANIMATION - ((NUM_INPUT_FRAMES + 1) * offset)))

    locations = []
    velocities = []

    for val_case in tqdm(val_cases[:100]):

        pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH,
                                                'pointset',
                                                f'force_{val_case[0]}',
                                                f'angle_{val_case[1]}',
                                                f'pos_{val_case[2]}_{val_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        for timestep in sequence_range:
            position_vector = [get_nested_pointset(pointsets[timestep + offset * j]) for j in range(NUM_INPUT_FRAMES + 1)]
            velocity_vector = get_velocity_from_position(position_vector, include_vel)
            locations.append(position_vector[:-1])
            velocities.append(velocity_vector)

    print('=========== Checking Val Data Type ===========')
    val_loc_shape = np.array(locations).shape
    val_vel_shape = np.array(velocities).shape

    print(val_loc_shape)
    print(val_vel_shape)

    savepath = create_directory(f'../preprocessed_data_dnri/offset_{offset}_input_{NUM_INPUT_FRAMES}')

    print('=========== Saving dNRI Val Data ===========')
    val_loc = np.array(locations)
    val_vel = np.array(velocities)

    if include_vel:
        np.save(os.path.join(savepath, 'loc_valid_mine'), val_loc)
        np.save(os.path.join(savepath, 'vel_valid_mine'), val_vel)
    else:
        np.save(os.path.join(savepath, 'loc_valid_mine_zeros'), val_loc)
        np.save(os.path.join(savepath, 'vel_valid_mine_zeros'), val_vel)

    return


def generate_test_data(test_cases, offset, include_vel):

    sequence_range = list(range(NUM_SEQUENCE_PER_ANIMATION - ((NUM_INPUT_FRAMES + 1) * offset)))

    locations = []
    velocities = []

    for test_case in tqdm(test_cases[:3]):

        pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH,
                                                'pointset',
                                                f'force_{test_case[0]}',
                                                f'angle_{test_case[1]}',
                                                f'pos_{test_case[2]}_{test_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        for timestep in sequence_range:
            position_vector = [get_nested_pointset(pointsets[timestep + offset * j]) for j in range(NUM_INPUT_FRAMES + 1)]
            velocity_vector = get_velocity_from_position(position_vector, include_vel)
            locations.append(position_vector[:-1])
            velocities.append(velocity_vector)

    print('=========== Checking Test Data Type ===========')
    test_loc_shape = np.array(locations).shape
    test_vel_shape = np.array(velocities).shape

    print(test_loc_shape)
    print(test_vel_shape)

    savepath = create_directory(f'../preprocessed_data_dnri/offset_{offset}_input_{NUM_INPUT_FRAMES}')

    print('=========== Saving dNRI Test Data ===========')
    test_loc = np.array(locations)
    test_vel = np.array(velocities)

    if include_vel:
        np.save(os.path.join(savepath, 'loc_test_mine'), test_loc)
        np.save(os.path.join(savepath, 'vel_test_mine'), test_vel)
    else:
        np.save(os.path.join(savepath, 'loc_test_mine_zeros'), test_loc)
        np.save(os.path.join(savepath, 'vel_test_mine_zeros'), test_vel)

    return


def generate_eval_data(offset, include_vel):

    locations = []
    velocities = []

    for test_case in tqdm(DNRI_TEST_CASES):

        pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH,
                                                'pointset',
                                                f'force_{test_case[0]}',
                                                f'angle_{test_case[1]}',
                                                f'pos_{test_case[2]}_{test_case[3]}',
                                                'ordered_unnormalized_pointset.json'))
        ptr.close()

        position_vector = [get_nested_pointset(pointsets[0 + offset * j]) for j in range(NUM_EVAL_FRAMES + 1)]
        velocity_vector = get_velocity_from_position_for_eval(position_vector, include_vel)
        locations.append(position_vector[:-1])
        velocities.append(velocity_vector)

    print('=========== Checking Test Data Type ===========')
    test_loc_shape = np.array(locations).shape
    test_vel_shape = np.array(velocities).shape

    print(test_loc_shape)
    print(test_vel_shape)

    savepath = create_directory(f'../preprocessed_data_dnri/offset_{offset}_input_{NUM_EVAL_FRAMES}')

    print('=========== Saving dNRI Eval Data ===========')
    test_loc = np.array(locations)
    test_vel = np.array(velocities)

    if include_vel:
        np.save(os.path.join(savepath, 'loc_eval_mine'), test_loc)
        np.save(os.path.join(savepath, 'vel_eval_mine'), test_vel)
    else:
        np.save(os.path.join(savepath, 'loc_eval_mine_zeros'), test_loc)
        np.save(os.path.join(savepath, 'vel_eval_mine_zeros'), test_vel)

    return
