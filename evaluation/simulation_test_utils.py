from tensorflow.keras.models import load_model
import numpy as np
from common.Constants import *
from common.Utils import load_json, normalize_pointset, get_nested_pointset, normalize_nested_pointset, denormalize_dnri_pointset, sort_pointset_by_ascending_x, sort_pointset_by_descending_y, center_transform
from tqdm import tqdm
from loss import get_cd_loss
from random import sample
import matplotlib.pyplot as plt
from box2d_simulator.wall_factory import *
from box2d_simulator.softbody_factory import *
import pygame
from Box2D.b2 import *
from pygame.locals import *


def visualize_synthetic_pointset(point_set, frame_size, wall_size):
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

    circleShape.draw = my_draw_circle
    polygonShape.draw = my_draw_polygon

    for body_number, body in enumerate(world.bodies):
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture, body_number)

    cropped = pygame.Surface((frame_size, frame_size))
    cropped.blit(screen, (0, 0), (wall_size + 1, wall_size + 1, frame_size, frame_size))
    image_numpy = pygame.surfarray.array2d(cropped)
    pygame.quit()
    return image_numpy.swapaxes(0, 1)


def generate_rollout_error_graph(args, savepath, errors):
    timestamps = [i for i in range(args.test_length)]
    plt.plot(timestamps, errors.tolist(), label=f'TP-Net-{args.num_input}', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel(f'Average {args.error_type} Error')
    plt.legend()
    plt.savefig(os.path.join(savepath, f'Average {args.error_type} Error - {args.env_name}_data.png'), dpi=600)
    plt.clf()

    print(f'Error at 40th timestep      : {round(float(errors[39]), 3)}')
    print(f'Error at 80th timestep      : {round(float(errors[79]), 3)}')


def preprocess_dnri_predictions(predictions):
    predictions = predictions[:, :, :, :, :2]             # discard predicted velocity
    for test_case in range(60):                           # for 60 simulation test cases
        for timestep in range(len(predictions[0][0])):    # change the metrics from -1~1 to 0~1
            predictions[test_case][0][timestep] = np.array(normalize_nested_pointset(denormalize_dnri_pointset(predictions[test_case][0][timestep])))

    return predictions.reshape(60, -1, 30, 2)


def get_error_for_sim_data(model_type, seed, num_input, test_length, error_type, data_type='ordered'):
    if model_type == 'static_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_static_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dynamic_nri':
        predictions = np.load(f'../result/nri/nri-{num_input}/seed_{seed}/softbody_predictions_dynamic_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dnri':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')
        predictions = preprocess_dnri_predictions(predictions)

    elif model_type == 'dynamics_prior':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')
        predictions = predictions[:, :, :, :2].tolist()

    elif model_type == 'graphrnn':
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions-num_samples_15_{data_type}.npy')
        predictions = predictions[:, :, :, :2]

    else:  # Ours (TP-Net)
        predictions = np.load(f'../result/{model_type}/{model_type}-{num_input}/seed_{seed}/softbody_predictions_{data_type}.npy')

    errors = []
    dnri_offset = 1
    if model_type == 'dnri':
        dnri_offset = 0

    if error_type == 'Position':
        for i, test_case in tqdm(list(enumerate(SIM_DATA_EVAL_CASES))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * SIM_DATA_OFFSET]], dtype='float32')
                errors[i].append(get_cd_loss(ground_truth, np.array([predictions[i][dnri_offset + timestep]], dtype='float32')))

    else:  # error_type == 'Shape'
        for i, test_case in tqdm(list(enumerate(SIM_DATA_EVAL_CASES))):
            errors.append([])
            ground_truth_pointsets = get_ground_truth_pointset(test_case)
            normalized_ground_truth_pointset = [normalize_nested_pointset(pointset) for pointset in ground_truth_pointsets]

            for timestep in range(test_length):
                ground_truth = np.array([normalized_ground_truth_pointset[(num_input + 1 + timestep) * SIM_DATA_OFFSET]], dtype='float32')
                transformed_ground_truth = center_transform(ground_truth)
                transformed_prediction = center_transform(np.array([predictions[i][dnri_offset + timestep]], dtype='float32'))
                errors[i].append(get_cd_loss(transformed_ground_truth, transformed_prediction))

    return errors


def load_pred_model(pred_model_path):
    print('================ Loading Model ================')
    return load_model(filepath=pred_model_path, compile=False)


def get_simulation_input_pointset(test_case_info, num_input, init_data_type):
    force, angle, init_x_pos, init_y_pos = test_case_info
    pointset, ptr = load_json(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()
    input_pointset = [normalize_pointset(pointset[i]) for i in range(0, SIM_DATA_OFFSET * num_input, SIM_DATA_OFFSET)]

    if init_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif init_data_type == 'sorted_x':
        input_pointset = [sort_pointset_by_ascending_x(pointset) for pointset in input_pointset]
    elif init_data_type == 'sorted_y':
        input_pointset = [sort_pointset_by_descending_y(pointset) for pointset in input_pointset]

    return np.array([input_pointset])


def get_ground_truth_pointset(test_case_info):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointsets, ptr = load_json(os.path.join(SIM_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    return [get_nested_pointset(pointset) for pointset in pointsets]
