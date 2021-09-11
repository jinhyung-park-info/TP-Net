from tensorflow.keras.models import load_model
import numpy as np
from common.Constants import *
from common.Utils import load_json, normalize_pointset, get_nested_pointset, sort_pointset, normalize_nested_pointset, denormalize_dnri_pointset
from tqdm import tqdm
from loss import get_cd_loss_func
from random import sample
from simulator.WallFactory import *
from simulator.SoftBodyFactory import *
import pygame
from Box2D.b2 import *
from pygame.locals import *


def preprocess_dnri_predictions(predictions):
    predictions = predictions[:, :, :, :, :2]  # discard velocity
    for test_case in range(100):
        for timestep in range(len(predictions[0][0])):
            predictions[test_case][0][timestep] = np.array(normalize_nested_pointset(denormalize_dnri_pointset(predictions[test_case][0][timestep])))

    return predictions.reshape(100, -1, 30, 2)



def center_transform(pointset):
    sum_info = np.sum(pointset[0], axis=0)
    center_x = sum_info[0] / 30
    center_y = sum_info[1] / 30
    for i in range(30):
        pointset[0][i][0] -= center_x
        pointset[0][i][1] -= center_y
    return pointset


def get_error(model_type, seed, num_input, test_length, test_animations, error_type, offset, start_timestep, num_samples):
    if model_type == 'static_nri':
        if start_timestep == 0:
            predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_static.npy')
        else:
            predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_static_{start_timestep}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dynamic_nri':
        if start_timestep == 0:
            predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_dynamic.npy')
        else:
            predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_dynamic_{start_timestep}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dnri':
        if start_timestep == 0:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions.npy')
        else:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{start_timestep}.npy')

        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dpi' or model_type == 'dpi_recursive':
        if start_timestep == 0:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions.npy')
        else:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{start_timestep}.npy')
        predictions = predictions[:, :, :, :2].tolist()

    elif model_type == 'graphrnn':
        if start_timestep == 0:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions-num_samples_{num_samples}.npy')
        else:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{start_timestep}-num_samples_{num_samples}.npy')
        predictions = predictions[:, :, :, :2]

    else:
        if start_timestep == 0:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions.npy')
        else:
            predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{start_timestep}.npy')

    errors = []
    dnri_correction = 1
    if model_type == 'dnri':
        dnri_correction = 0

    if error_type == 'Position':
        for i, test_case in tqdm(list(enumerate(EVAL_CASES[:test_animations]))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + start_timestep + 1 + timestep) * offset]], dtype='float32')
                errors[i].append(get_cd_loss_func(ground_truth, np.array([predictions[i][dnri_correction + timestep]], dtype='float32')))

    else:
        for i, test_case in tqdm(list(enumerate(EVAL_CASES[:test_animations]))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + start_timestep + timestep) * offset]], dtype='float32')
                transformed_ground_truth = center_transform(ground_truth)
                transformed_prediction = center_transform(np.array([predictions[i][dnri_correction + timestep]], dtype='float32'))
                errors[i].append(get_cd_loss_func(transformed_ground_truth, transformed_prediction))

    return errors


def draw_box2d_image(point_set):
    screen = pygame.display.set_mode((SCREEN_WD, SCREEN_HT), 0, 32)
    world = b2World(gravity=(0, 0), doSleep=False)
    WallFactory(world, SCREEN_LENGTH, SCREEN_HEIGHT, WALL_WIDTH).create_walls()

    def my_draw_polygon(polygon, body, fixture, body_number):
        vertices = [(body.transform * v) * PPM for v in polygon.vertices]
        vertices = [(v[0], SCREEN_HT - v[1]) for v in vertices]
        pygame.draw.polygon(screen, body_number_to_color[body_number], vertices)

    def my_draw_circle(circle, body, fixture, body_number):
        position = body.transform * circle.pos * PPM
        position = (position[0], SCREEN_HT - position[1])
        pygame.draw.circle(screen, body_number_to_color[body_number], [int(x) for x in position], int(circle.radius * PPM))

    """
    for i in range(NUM_PARTICLES):
        sat_body_def = b2BodyDef()
        sat_body_def.type = b2_staticBody
        sat_body_def.position.Set(point_set[i][0], point_set[i][1])
        sat_body = world.CreateBody(sat_body_def)
        sat_body_shape = b2CircleShape(radius=SATELLITE_RADIUS * 3)
        sat_body_fixture = b2FixtureDef(shape=sat_body_shape, density=SOFTBODY_DENSITY, friction=SOFTBODY_FRICTION, restitution=SOFTBODY_RESTITUTION)
        sat_body_fixture.filter.categoryBits = SOFTBODYBITS
        sat_body_fixture.filter.maskBits = 0x0001 | SOFTBODYBITS
        sat_body.CreateFixture(sat_body_fixture)
    """
    for i in range(NUM_PARTICLES):
        sat_body_def = b2BodyDef()
        sat_body_def.type = b2_staticBody
        sat_body_def.position.Set(point_set[i][0], point_set[i][1])
        sat_body = world.CreateBody(sat_body_def)
        sat_body_shape = b2CircleShape(radius=SATELLITE_RADIUS * 2)
        sat_body_fixture = b2FixtureDef(shape=sat_body_shape, density=SOFTBODY_DENSITY, friction=SOFTBODY_FRICTION, restitution=SOFTBODY_RESTITUTION)
        sat_body_fixture.filter.categoryBits = SOFTBODYBITS
        sat_body_fixture.filter.maskBits = 0x0001 | SOFTBODYBITS
        sat_body.CreateFixture(sat_body_fixture)

    circleShape.draw = my_draw_circle
    polygonShape.draw = my_draw_polygon

    for body_number, body in enumerate(world.bodies):
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture, body_number)

    cropped = pygame.Surface((860, 860))
    cropped.blit(screen, (0, 0), (21, 21, 860, 860))
    image_numpy = pygame.surfarray.array2d(cropped)
    pygame.quit()
    return image_numpy.swapaxes(0, 1)


def load_pred_model(pred_model_path):
    print('================ Loading Model ================')
    return load_model(filepath=pred_model_path, compile=False)


def get_simulation_input_pointset(test_case_info, offset, test_data_type, num_input, start_timestep=0):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointset, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    # Prepare initial input for predicting this animation
    input_pointset = [normalize_pointset(pointset[i]) for i in range(offset * start_timestep, offset * (start_timestep + num_input), offset)]
    if test_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif test_data_type == 'sorted':
        input_pointset = [sort_pointset(pointset) for pointset in input_pointset]

    return np.array([input_pointset])  # shape == (1, NUM_INPUT_FRAMES, 30, 2)


def update_input_pointset(old_input, predicted_pointset):
    return np.expand_dims(np.concatenate([old_input[0][1:], predicted_pointset], axis=0), axis=0)  # shape == (1, NUM_INPUT_FRAMES, 20, 2)


def get_ground_truth_pointset(test_case_info):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    # Prepare initial input for predicting this animation
    return [get_nested_pointset(pointset) for pointset in pointsets]


def get_normalized_ground_truth_pointset(test_case_info):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointsets, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    return [normalize_pointset(pointset) for pointset in pointsets]
