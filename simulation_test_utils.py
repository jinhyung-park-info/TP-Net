from tensorflow.keras.models import load_model
import numpy as np
from common.Constants import *
from common.Utils import load_json, normalize_pointset, create_directory, denormalize_pointset, get_nested_pointset, sort_pointset
from tqdm import tqdm
from PIL import Image
from loss import get_cd_loss_func
import matplotlib.pyplot as plt
from Box2D.b2 import *
from simulator.WallFactory import *
from simulator.SoftBodyFactory import *
import pygame
from pygame.locals import *
from random import sample


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

    for i in range(NUM_PARTICLES):
        sat_body_def = b2BodyDef()
        sat_body_def.type = b2_staticBody
        sat_body_def.position.Set(point_set[i][0], point_set[i][1])
        sat_body = world.CreateBody(sat_body_def)
        sat_body_shape = b2CircleShape(radius=SATELLITE_RADIUS)
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


def get_simulation_input_pointset(test_case_info, offset, test_data_type):
    force = test_case_info[0]
    angle = test_case_info[1]
    init_x_pos = test_case_info[2]
    init_y_pos = test_case_info[3]

    pointset, ptr = load_json(os.path.join(RAW_DATA_PATH, 'pointset', f'force_{force}', f'angle_{angle}', f'pos_{init_x_pos}_{init_y_pos}', 'ordered_unnormalized_pointset.json'))
    ptr.close()

    # Prepare initial input for predicting this animation
    input_pointset = [normalize_pointset(pointset[i]) for i in range(0, offset * NUM_INPUT_FRAMES, offset)]
    if test_data_type == 'unordered':
        input_pointset = [sample(pointset, len(pointset)) for pointset in input_pointset]
    elif test_data_type == 'sorted':
        input_pointset = [sort_pointset(pointset) for pointset in input_pointset]

    return np.array([input_pointset])  # shape == (1, NUM_INPUT_FRAMES, 30, 2)


def update_input_pointset(old_input, predicted_pointset):
    return np.expand_dims(np.concatenate([old_input[0][1:], predicted_pointset], axis=0), axis=0) # shape == (1, NUM_INPUT_FRAMES, 20, 2)


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

    # Prepare initial input for predicting this animation
    return [normalize_pointset(pointset) for pointset in pointsets]


def make_and_concat_two_simulation_ground_truth_frames(pointset, savepath, sequence_number):
    merged = Image.new(mode="L", size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=255)
    ground_truth_image = Image.fromarray(draw_box2d_image(pointset))
    merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image, box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(savepath, f'timestep_{sequence_number}.jpg'))
    return np.array(merged)


def make_and_concat_pred_and_simulation_gt_frame(predicted_frame, pointset, sequence_number, savepath):
    merged = Image.new(mode="L", size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=255)
    merged.paste(im=Image.fromarray(draw_box2d_image(pointset)), box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=Image.fromarray(predicted_frame), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(savepath, f'timestep_{sequence_number}.jpg'))
    return np.array(merged)


def generate_simulation_result_videos(prediction_model, save_path, offset, length, fps, test_type, test_data_type):

    if test_type == 'train':
        cases = TEST_CASES[:15]
    else:
        cases = TEST_CASES[:15]

    for case_num, test_case in enumerate(cases):

        print(f'============== Testing Case #{case_num + 1} ==============')
        num_predicted = 0
        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case_num + 1}.MP4'), CODEC, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case_num + 1}'))
        input_info = get_simulation_input_pointset(test_case, offset, test_data_type)

        pointsets = get_ground_truth_pointset(test_case)

        frame = make_and_concat_two_simulation_ground_truth_frames(pointsets[0], frames_savepath, sequence_number=0)
        for _ in range(20):
            output_video.write(frame)

        for timestep in range(0, NUM_INPUT_FRAMES * offset, offset):
            frame = make_and_concat_two_simulation_ground_truth_frames(pointsets[timestep], frames_savepath, timestep)
            output_video.write(frame)

        for timestep in tqdm(range(offset * NUM_INPUT_FRAMES, offset * (NUM_INPUT_FRAMES + length), offset)):

            predicted_pointset = prediction_model.predict(input_info)[0]

            assert predicted_pointset.shape == (1, NUM_PARTICLES, 2), 'Expected {} but received {}'.format((1, NUM_PARTICLES, 2), predicted_pointset.shape)

            coordinates = denormalize_pointset(predicted_pointset[0])
            predicted_frame = draw_box2d_image(coordinates)

            # concatenate with ground truth image for comparison
            merged_frame = make_and_concat_pred_and_simulation_gt_frame(predicted_frame, pointsets[timestep], timestep, frames_savepath)
            output_video.write(merged_frame)
            num_predicted += 1

            input_info = update_input_pointset(input_info, predicted_pointset)

        output_video.release()
        cv2.destroyAllWindows()


# Compare Multiple Simulation Prediction
"""
def compare_simulation_prediction_results(model_1, model_2, model_1_type, model_2_type, save_path, offset, fps, test_type, test_data_type):

    if test_data_type == 'unordered':
        if test_type == 'train':
            cases = TRAINED_CASES_FOR_UNORDERED
        else:
            cases = TEST_CASES_FOR_UNORDERED
    else:
        if test_type == 'train':
            cases = TRAINED_CASES_FOR_ORDERED
        else:
            cases = TEST_CASES_FOR_ORDERED

    global_losses_1 = np.array([0.0] * 72)
    global_losses_2 = np.array([0.0] * 72)

    for case_num, test_case in enumerate(cases):

        print(f'============== Testing Case #{case_num + 1} ==============')

        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case_num + 1}.MP4'), CODEC, fps, (VIDEO_HEIGHT * 3, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case_num + 1}'))
        ground_truth_base_path = get_ground_truth_base_path(test_case)
        input_info_1 = get_simulation_input_pointset(test_case, offset, test_data_type)
        input_info_2 = get_simulation_input_pointset(test_case, offset, test_data_type)
        ground_truth_pointset = get_normalized_ground_truth_pointset(test_case)

        timesteps = []
        losses_1 = []
        losses_2 = []

        frame = concat_three_simulation_ground_truth_frames(ground_truth_base_path, frames_savepath, sequence_number=0)
        for i in range(20):
            output_video.write(frame)

        for timestep in range(0, NUM_INPUT_FRAMES * offset, offset):
            frame = concat_three_simulation_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep)
            output_video.write(frame)

        for timestep in tqdm(range(offset * NUM_INPUT_FRAMES, NUM_SEQUENCE_PER_ANIMATION, offset)):
            predicted_pointset_1 = model_1.predict(input_info_1)[0]     # Use only the first predicted frame
            predicted_pointset_2 = model_2.predict(input_info_2)[0]

            loss_1 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_1)
            loss_2 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_2)

            timesteps.append(timestep)
            losses_1.append(loss_1)
            losses_2.append(loss_2)

            coordinates_1 = denormalize_pointset(predicted_pointset_1[0])
            coordinates_2 = denormalize_pointset(predicted_pointset_2[0])
            predicted_frame_1 = draw_box2d_image(coordinates_1)
            predicted_frame_2 = draw_box2d_image(coordinates_2)

            # concatenate with ground truth image for comparison
            merged_frame = concat_multiple_preds_and_simulation_gt_frame(predicted_frame_1, predicted_frame_2, ground_truth_base_path, timestep, frames_savepath)
            output_video.write(merged_frame)
            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)

        output_video.release()
        cv2.destroyAllWindows()

        plt.plot(timesteps, losses_1, label=f'{model_1_type}')
        plt.plot(timesteps, losses_2, label=f'{model_2_type}')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_test_case_{}.png'.format(case_num + 1)), dpi=600)
        plt.clf()

        global_losses_1 += losses_1
        global_losses_2 += losses_2

    # Average Loss Graph
    global_losses_1 /= len(cases)
    global_losses_2 /= len(cases)
    plt.plot(timesteps, global_losses_1.tolist(), label=f'{model_1_type}')
    plt.plot(timesteps, global_losses_2.tolist(), label=f'{model_2_type}')
    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average.png'), dpi=600)
    plt.clf()

    plt.plot(timesteps[:30], global_losses_1.tolist()[:30], label=f'{model_1_type}')
    plt.plot(timesteps[:30], global_losses_2.tolist()[:30], label=f'{model_2_type}')
    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average_first_30_frames.png'), dpi=600)
    plt.clf()
"""

def make_and_concat_four_simulation_ground_truth_frames(pointset, savepath, sequence_number):
    merged = Image.new(mode="L", size=(SMALL_HEIGHT * 4, SMALL_HEIGHT), color=255)
    ground_truth_image = Image.fromarray(draw_box2d_image(pointset)).resize((SMALL_SIZE, SMALL_SIZE))
    merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image, box=(SMALL_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image, box=(SMALL_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image, box=(SMALL_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(savepath, f'timestep_{sequence_number}.jpg'))
    return np.array(merged)


def make_and_concat_three_preds_and_simulation_gt_frame(predicted_frame_1, predicted_frame_2, predicted_frame_3, pointset, sequence_number, savepath):
    merged = Image.new(mode="L", size=(SMALL_HEIGHT * 4, SMALL_HEIGHT), color=255)
    merged.paste(im=Image.fromarray(draw_box2d_image(pointset)).resize((SMALL_SIZE, SMALL_SIZE)), box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=Image.fromarray(predicted_frame_1).resize((SMALL_SIZE, SMALL_SIZE)), box=(SMALL_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.paste(im=Image.fromarray(predicted_frame_2).resize((SMALL_SIZE, SMALL_SIZE)), box=(SMALL_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
    merged.paste(im=Image.fromarray(predicted_frame_3).resize((SMALL_SIZE, SMALL_SIZE)), box=(SMALL_HEIGHT * 3 + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(savepath, f'timestep_{sequence_number}.jpg'))
    return np.array(merged)


def compare_baseline_and_ours_prediction(model_1, model_2, model_3, data_1_type, data_2_type, data_3_type, save_path, offset, fps):

    cases = TEST_CASES[:15]
    global_losses_1 = np.array([0.0] * 145)
    global_losses_2 = np.array([0.0] * 145)
    global_losses_3 = np.array([0.0] * 145)

    for case_num, test_case in enumerate(cases):

        print(f'============== Testing Case #{case_num + 1} ==============')

        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case_num + 1}.MP4'), CODEC, fps, (VIDEO_HEIGHT * 3, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case_num + 1}'))

        input_info_1 = get_simulation_input_pointset(test_case, offset, data_1_type)
        input_info_2 = get_simulation_input_pointset(test_case, offset, data_2_type)
        input_info_3 = get_simulation_input_pointset(test_case, offset, data_3_type)

        ground_truth_pointset = get_ground_truth_pointset(test_case)
        normalized_ground_truth_pointset = get_normalized_ground_truth_pointset(test_case)

        timesteps = []
        losses_1 = []
        losses_2 = []
        losses_3 = []

        frame = make_and_concat_four_simulation_ground_truth_frames(ground_truth_pointset[0], frames_savepath, 0)
        for i in range(20):
            output_video.write(frame)

        for timestep in range(0, NUM_INPUT_FRAMES * offset, offset):
            frame = make_and_concat_four_simulation_ground_truth_frames(ground_truth_pointset[timestep], frames_savepath, timestep)
            output_video.write(frame)

        for timestep in tqdm(range(offset * NUM_INPUT_FRAMES, NUM_SEQUENCE_PER_ANIMATION, offset)):
            predicted_pointset_1 = model_1.predict(input_info_1)[0]
            predicted_pointset_2 = model_2.predict(input_info_2)[0]
            predicted_pointset_3 = model_3.predict(input_info_3)[0]

            loss_1 = get_cd_loss_func(np.array([normalized_ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_1)
            loss_2 = get_cd_loss_func(np.array([normalized_ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_2)
            loss_3 = get_cd_loss_func(np.array([normalized_ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_3)

            timesteps.append(timestep)
            losses_1.append(loss_1)
            losses_2.append(loss_2)
            losses_3.append(loss_3)

            coordinates_1 = denormalize_pointset(predicted_pointset_1[0])
            coordinates_2 = denormalize_pointset(predicted_pointset_2[0])
            coordinates_3 = denormalize_pointset(predicted_pointset_3[0])

            predicted_frame_1 = draw_box2d_image(coordinates_1)
            predicted_frame_2 = draw_box2d_image(coordinates_2)
            predicted_frame_3 = draw_box2d_image(coordinates_3)

            # concatenate with ground truth image for comparison
            merged_frame = make_and_concat_three_preds_and_simulation_gt_frame(predicted_frame_1, predicted_frame_2, predicted_frame_3, ground_truth_pointset[timestep], timestep, frames_savepath)
            output_video.write(merged_frame)
            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)
            input_info_3 = update_input_pointset(input_info_3, predicted_pointset_3)

        output_video.release()
        cv2.destroyAllWindows()

        plt.plot(timesteps, losses_1, label=f'{data_1_type}')
        plt.plot(timesteps, losses_2, label=f'{data_2_type}')
        plt.plot(timesteps, losses_3, label=f'{data_3_type}')

        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_test_case_{}.png'.format(case_num + 1)), dpi=600)
        plt.clf()

        global_losses_1 += losses_1[:145]
        global_losses_2 += losses_2[:145]
        global_losses_3 += losses_3[:145]

    # Average Loss Graph
    global_losses_1 /= len(cases)
    global_losses_2 /= len(cases)
    global_losses_3 /= len(cases)

    plt.plot(timesteps[:145], global_losses_1.tolist(), label=f'{data_1_type}')
    plt.plot(timesteps[:145], global_losses_2.tolist(), label=f'{data_2_type}')
    plt.plot(timesteps[:145], global_losses_3.tolist(), label=f'{data_3_type}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average.png'), dpi=600)
    plt.clf()

    plt.plot(timesteps[:30], global_losses_1.tolist()[:30], label=f'{data_1_type}')
    plt.plot(timesteps[:30], global_losses_2.tolist()[:30], label=f'{data_2_type}')
    plt.plot(timesteps[:30], global_losses_3.tolist()[:30], label=f'{data_3_type}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average_first_30_frames.png'), dpi=600)
    plt.clf()
