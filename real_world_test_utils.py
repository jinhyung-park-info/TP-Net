import numpy as np
from common.Constants import *
from common.Utils import load_json, create_directory, denormalize_pointset, sort_pointset
from tqdm import tqdm
from PIL import Image
from simulation_test_utils import update_input_pointset
import re
from Box2D.b2 import *
from simulator.WallFactory import *
from simulator.SoftBodyFactory import *
import pygame
from pygame.locals import *
from random import sample
from loss import get_cd_loss_func
import matplotlib.pyplot as plt

VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 900
FRAME_SIZE = 860
CROP_SIZE = 922
WALL_SIZE = int((VIDEO_HEIGHT - FRAME_SIZE) / 2)


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


def get_real_world_input_pointset(case, num_input_frames, offset, test_data_type):
    input_info, ptr = load_json(os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{case}', 'ordered_normalized_state_vectors.json'))
    ptr.close()
    input_info = [input_info[offset * i] for i in range(num_input_frames)]

    if test_data_type == 'unordered':
        input_info = [sample(pointset, len(pointset)) for pointset in input_info]

    return np.array([input_info])


def find_case_info(path):
    directories = os.listdir(path)
    invalid_names = ['output.mp4', 'output.MP4']
    for dirname in invalid_names:
        if dirname in directories:
            directories.remove(dirname)
    num_frames = len(directories)
    directories = sorted(directories, key=lambda file: int(re.findall('\d+', file)[0]))

    first_frame_number = directories[0][9:][:-4]
    return int(first_frame_number), num_frames


def concat_two_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep, first_frame_number, save):
    ground_truth_image = Image.open(os.path.join(ground_truth_base_path, f'timestep_{first_frame_number + timestep}.jpg')).resize((FRAME_SIZE, FRAME_SIZE))
    merged = Image.new(mode="L", size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=255)
    merged.paste(im=ground_truth_image.resize((FRAME_SIZE, FRAME_SIZE)), box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image.resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    if save:
        merged.save(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'))
    return np.array(merged)


def concat_pred_and_real_world_gt_frame(predicted_frame, ground_truth_base_path, timestep, save_path, first_frame_number, num_frames):
    merged = Image.new(mode="L", size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=255)

    # Paste Ground Truth Frame on the left
    if timestep < num_frames:
        ground_truth_path = os.path.join(ground_truth_base_path, f'timestep_{first_frame_number + timestep}.jpg')
        ground_truth_image = Image.open(ground_truth_path).resize((FRAME_SIZE, FRAME_SIZE))
        merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))

    # Paste Predicted Frame on the right
    merged.paste(im=Image.fromarray(predicted_frame).resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(save_path, f'timestep_{timestep}.jpg'))
    return np.array(merged)


def generate_real_result_videos(prediction_model, num_input, save_path, offset, fps, test_data_type):

    for case in REAL_WORLD_TEST_CASES:

        print(f'============== Testing Case #{case} ==============')

        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case}.MP4'), CODEC, fps, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case_{case}'))
        ground_truth_base_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{case}')
        input_info = get_real_world_input_pointset(case, num_input, offset, test_data_type)

        first_frame_number, num_frames = find_case_info(ground_truth_base_path)

        frame = concat_two_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, 0, first_frame_number, save=False)
        for _ in range(20):
            output_video.write(frame)

        for timestep in range(0, num_input * offset, offset):
            frame = concat_two_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep, first_frame_number, save=True)
            output_video.write(frame)

        for timestep in tqdm(range(offset * num_input, num_frames, offset)):
            predicted_pointset = prediction_model.predict(input_info)[0]         # Use only the first predicted frame

            assert predicted_pointset.shape == (1, NUM_PARTICLES, 2), 'Expected {} but received {}'.format((1, NUM_PARTICLES, 2), predicted_pointset.shape)
            coordinates = denormalize_pointset(predicted_pointset[0])
            predicted_frame = draw_box2d_image(coordinates)

            # concatenate with ground truth image for comparison
            merged_frame = concat_pred_and_real_world_gt_frame(predicted_frame, ground_truth_base_path, timestep, frames_savepath, first_frame_number, num_frames)
            output_video.write(merged_frame)
            input_info = update_input_pointset(input_info, predicted_pointset)

        output_video.release()
        cv2.destroyAllWindows()


def get_real_world_ground_truth_pointset(case, test_data_type):
    path = os.path.join(REAL_DATA_PATH, '05_postprocessed_data', f'case_{case}', 'ordered_normalized_state_vectors.json')
    pointsets, ptr = load_json(path)
    ptr.close()

    if test_data_type == 'unordered':
        pointsets = [sample(pointset, len(pointset)) for pointset in pointsets]
    elif test_data_type == 'sorted':
        pointsets = [sort_pointset(pointset) for pointset in pointsets]

    return pointsets


def sort_clockwise(pointset):

    sorted_pointset = sorted(pointset, key=lambda coord: (-coord[1], coord[0]))
    top = sorted_pointset[0]
    bottom = sorted_pointset[-1]

    right_points = []
    left_points = []
    sorted_pointset.remove(top)
    sorted_pointset.remove(bottom)
    for point in sorted_pointset:
        if point[0] >= top[0] and point[0] >= bottom[0]:
            assert bottom[1] <= point[1] <= top[1]
            right_points.append(point)
        else:
            left_points.append(point)

    right_points = sorted(right_points, key=lambda coord: (-coord[1], coord[0]))
    left_points = sorted(left_points, key=lambda coord: (coord[1], -coord[0]))
    sorted_pointset = [top]
    sorted_pointset.extend(right_points)
    sorted_pointset.append(bottom)
    sorted_pointset.extend(left_points)
    return np.array(sorted_pointset)


def determine_center_from_sparse_contour(sparse_pointset):
    M = cv2.moments(sparse_pointset)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return center_x, center_y


def draw_polygon(ImShape, Polygon, Color):
    Im = np.zeros(ImShape, np.uint8)
    try:
        cv2.fillPoly(Im, Polygon, Color) # Only using this function may cause errors, I don’t know why
    except:
        try:
            cv2.fillConvexPoly(Im, Polygon, Color)
        except:
            print('cant fill\n')

    return Im


def get_non_intersected_area(image_shape, polygon_1, polygon_2):
    image_1 = draw_polygon(image_shape[:-1], polygon_1, 133)           # Polygon 2 area is filled with 133
    ret, new_image_1 = cv2.threshold(image_1, 120, 133, cv2.THRESH_BINARY)   # According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
    ground_truth_area = np.sum(np.greater(new_image_1, 0))                   # Find the area of the overlapping area of two polygons

    image_2 = draw_polygon(image_shape[:-1], polygon_2, 133)             # Polygon 2 area is filled with 133
    ret, new_image_2 = cv2.threshold(image_2, 120, 133, cv2.THRESH_BINARY)     # According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
    predicted_area = np.sum(np.greater(new_image_2, 0))                        # Find the area of the overlapping area of two polygons

    image_1 = draw_polygon(image_shape[:-1], polygon_1, 122)
    image_2 = draw_polygon(image_shape[:-1], polygon_2, 133)
    image = image_1 + image_2
    ret, overlapped_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)   # According to the above filling value, so the pixel value in the new image is 255 as the overlapping place
    intersect_area = np.sum(np.greater(overlapped_image, 0))                    # Find the area of the overlapping area of two polygons

    return (predicted_area - intersect_area) + (ground_truth_area - intersect_area)


def get_area_loss(ground_truth_pointset, predicted_pointset):

    ground_truth_pixel_pointset = []
    for x, y in ground_truth_pointset:
        pixel_x = int(x * FRAME_SIZE)
        pixel_y = int(FRAME_SIZE * (1 - y))
        ground_truth_pixel_pointset.append([pixel_x, pixel_y])

    predicted_pixel_pointset = []
    for x, y in predicted_pointset:
        pixel_x = int(x * FRAME_SIZE)
        pixel_y = int(FRAME_SIZE * (1 - y))
        predicted_pixel_pointset.append([pixel_x, pixel_y])

    ground_truth_pixel_pointset = sort_clockwise(ground_truth_pixel_pointset)
    predicted_pixel_pointset = sort_clockwise(predicted_pixel_pointset)

    gt_center_x, gt_center_y = determine_center_from_sparse_contour(ground_truth_pixel_pointset)
    pd_center_x, pd_center_y = determine_center_from_sparse_contour(predicted_pixel_pointset)

    gt_origin_transformation_matrix = np.array([-gt_center_x, -gt_center_y] * NUM_PARTICLES).reshape(NUM_PARTICLES, 2)
    pd_origin_transformation_matrix = np.array([-pd_center_x, -pd_center_y] * NUM_PARTICLES).reshape(NUM_PARTICLES, 2)
    middle_transformation_matrix = np.array([int(FRAME_SIZE / 2), int(FRAME_SIZE / 2)] * NUM_PARTICLES).reshape(NUM_PARTICLES, 2)

    ground_truth_pixel_pointset = ground_truth_pixel_pointset + gt_origin_transformation_matrix + middle_transformation_matrix
    predicted_pixel_pointset = predicted_pixel_pointset + pd_origin_transformation_matrix + middle_transformation_matrix

    contours = [ground_truth_pixel_pointset, predicted_pixel_pointset]
    non_intersected_area = get_non_intersected_area((FRAME_SIZE, FRAME_SIZE, 3), contours[0], contours[1])
    non_intersected_area = non_intersected_area / (FRAME_SIZE * FRAME_SIZE)

    return non_intersected_area


def generate_fine_tuning_result_videos(simulation_model, real_model, num_input_frames, save_path, offset, fps, test_data_type):

    global_losses_1 = np.array([0.0] * COMPARE_LENGTH)
    global_losses_2 = np.array([0.0] * COMPARE_LENGTH)

    global_area_losses_1 = np.array([0.0] * COMPARE_LENGTH)
    global_area_losses_2 = np.array([0.0] * COMPARE_LENGTH)

    for case in REAL_WORLD_TEST_CASES:

        print(f'=========== Testing Case #{case} =============')

        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case}.MP4'), CODEC, fps, (VIDEO_HEIGHT * 3, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case}'))
        ground_truth_base_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{case}')

        ground_truth_pointset = get_real_world_ground_truth_pointset(case, test_data_type)

        input_info_1 = get_real_world_input_pointset(case, num_input_frames, offset, test_data_type)
        input_info_2 = get_real_world_input_pointset(case, num_input_frames, offset, test_data_type)
        first_frame_number, num_frames = find_case_info(ground_truth_base_path)

        timesteps = []
        losses_1 = []
        losses_2 = []
        area_losses_1 = []
        area_losses_2 = []

        frame = concat_three_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, 0, first_frame_number)
        for _ in range(20):
            output_video.write(frame)

        for timestep in range(0, num_input_frames * offset, offset):
            frame = concat_three_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep, first_frame_number)
            output_video.write(frame)

        for timestep in tqdm(range(offset * num_input_frames, num_frames, offset)):
            predicted_pointset_1 = real_model.predict(input_info_1)[0]     # Use only the first predicted frame
            predicted_pointset_2 = simulation_model.predict(input_info_2)[0]

            area_loss_1 = get_area_loss(np.array(ground_truth_pointset[timestep], dtype='float32'), predicted_pointset_1[0])
            area_loss_2 = get_area_loss(np.array(ground_truth_pointset[timestep], dtype='float32'), predicted_pointset_2[0])

            loss_1 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_1)
            loss_2 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_2)

            timesteps.append(timestep)
            losses_1.append(loss_1)
            losses_2.append(loss_2)

            area_losses_1.append(area_loss_1)
            area_losses_2.append(area_loss_2)

            coordinates_1 = denormalize_pointset(predicted_pointset_1[0])
            coordinates_2 = denormalize_pointset(predicted_pointset_2[0])
            predicted_frame_1 = draw_box2d_image(coordinates_1)
            predicted_frame_2 = draw_box2d_image(coordinates_2)

            # concatenate with ground truth image for comparison
            merged_frame = concat_multiple_preds_and_real_world_gt_frame(predicted_frame_1, predicted_frame_2, ground_truth_base_path, timestep, frames_savepath, first_frame_number, num_frames)
            output_video.write(merged_frame)
            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)

        output_video.release()
        cv2.destroyAllWindows()

        plt.plot(timesteps, losses_1, label=f'before fine tuning')
        plt.plot(timesteps, losses_2, label=f'after fine tuning')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_test_case_{}.png'.format(case)), dpi=600)
        plt.clf()

        plt.plot(timesteps, area_losses_1, label=f'before fine tuning')
        plt.plot(timesteps, area_losses_2, label=f'after fine tuning')
        plt.xlabel('Timestep')
        plt.ylabel('Area Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'area_loss_test_case_{}.png'.format(case)), dpi=600)
        plt.clf()

        global_losses_1 += np.array(losses_1[:COMPARE_LENGTH])
        global_losses_2 += np.array(losses_2[:COMPARE_LENGTH])

        global_area_losses_1 += np.array(area_losses_1[:COMPARE_LENGTH])
        global_area_losses_2 += np.array(area_losses_2[:COMPARE_LENGTH])

    # Average Loss Graph
    global_losses_1 /= len(REAL_WORLD_TEST_CASES)
    global_losses_2 /= len(REAL_WORLD_TEST_CASES)
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_1.tolist(), label=f'before fine tuning')
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_2.tolist(), label=f'after fine tuning')
    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average.png'), dpi=600)
    plt.clf()

    plt.plot(timesteps[:30], global_losses_1.tolist()[:30], label=f'before fine tuning')
    plt.plot(timesteps[:30], global_losses_2.tolist()[:30], label=f'after fine tuning')
    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_average_first_30_frames.png'), dpi=600)
    plt.clf()

    # Average Area Loss Graph
    global_area_losses_1 /= len(REAL_WORLD_TEST_CASES)
    global_area_losses_2 /= len(REAL_WORLD_TEST_CASES)
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_1.tolist(), label=f'before fine tuning')
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_2.tolist(), label=f'after fine tuning')
    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'area_loss_average.png'), dpi=600)
    plt.clf()

    plt.plot(timesteps[:30], global_area_losses_1.tolist()[:30], label=f'before fine tuning')
    plt.plot(timesteps[:30], global_area_losses_2.tolist()[:30], label=f'after fine tuning')
    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'area_loss_average_first_30_frames.png'), dpi=600)
    plt.clf()

# Compare Multiple Real World Prediction

def concat_three_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep, first_frame_number):
    ground_truth_image = Image.open(os.path.join(ground_truth_base_path, f'timestep_{first_frame_number + timestep}.jpg')).resize((FRAME_SIZE, FRAME_SIZE))
    merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 3, VIDEO_HEIGHT), color=255)
    merged.paste(im=ground_truth_image.resize((FRAME_SIZE, FRAME_SIZE)), box=(WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image.resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.paste(im=ground_truth_image.resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'))
    return np.array(merged)


def concat_multiple_preds_and_real_world_gt_frame(predicted_frame_1, predicted_frame_2, ground_truth_base_path, timestep, save_path, first_frame_number, num_frames):
    merged = Image.new(mode="L", size=(VIDEO_HEIGHT * 3, VIDEO_HEIGHT), color=255)

    # Paste Ground Truth Frame on the left
    if timestep < num_frames:
        ground_truth_path = os.path.join(ground_truth_base_path, f'timestep_{first_frame_number + timestep}.jpg')
        ground_truth_image = Image.open(ground_truth_path).resize((FRAME_SIZE, FRAME_SIZE))
        merged.paste(im=ground_truth_image, box=(WALL_SIZE, WALL_SIZE))

    # Paste Predicted Frame on the right
    merged.paste(im=Image.fromarray(predicted_frame_1).resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT + WALL_SIZE, WALL_SIZE))
    merged.paste(im=Image.fromarray(predicted_frame_2).resize((FRAME_SIZE, FRAME_SIZE)), box=(VIDEO_HEIGHT * 2 + WALL_SIZE, WALL_SIZE))
    merged.save(os.path.join(save_path, f'timestep_{timestep}.jpg'))
    return np.array(merged)


def get_final_video_size(video_number):
    if video_number == 4:
        return 1624, 948
    src1 = cv2.imread(os.path.join(f'C:/Users/User/Desktop/crop_positions/case_{video_number}_left.jpg'))
    src2 = cv2.imread(os.path.join(f'C:/Users/User/Desktop/crop_positions/case_{video_number}_down.jpg'))
    return src1.shape[1], src2.shape[0]


def convert_to_pixel_coordinates(predicted_pointset, distance, height):
    pixel_pointset = []

    for point in predicted_pointset:
        x = int(CROP_SIZE * point[0])
        y = int(CROP_SIZE * (1 - point[1]))
        pixel_pointset.append([x, y])

    for i in range(NUM_PARTICLES):
        pixel_pointset[i][0] += 228
        pixel_pointset[i][1] += height - CROP_SIZE

    return sort_clockwise(pixel_pointset)


def generate_rendered_videos(simulation_model, real_model, num_input_frames, save_path, offset, fps, test_data_type):

    for case in REAL_WORLD_TEST_CASES:

        print(f'=========== Testing Case #{case} =============')
        distance, height = get_final_video_size(case)
        video_size = (575 * 3 + 80, 520)
        # video_size = ((228 + CROP_SIZE) * 2 + 50, 1040)
        # 1150 * 2 + 50
        # 1040
        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case}.MP4'), CODEC, fps, video_size)

        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case}'))
        background_image_1 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
        background_image_2 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]

        ground_truth_base_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{case}')

        input_info_1 = get_real_world_input_pointset(case, num_input_frames, offset, test_data_type)
        input_info_2 = get_real_world_input_pointset(case, num_input_frames, offset, test_data_type)
        first_frame_number, num_frames = find_case_info(ground_truth_base_path)

        ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
        ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

        merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
        merged.paste(im=ground_truth_image, box=(0, 0))
        merged.paste(im=ground_truth_image, box=(575 + 40, 0))
        merged.paste(im=ground_truth_image, box=(575 * 2 + 80, 0))
        merged = np.array(merged)
        cv2.imwrite(os.path.join(frames_savepath, f'timestep_{0}.jpg'), merged)

        for i in range(20):
            output_video.write(merged)

        for timestep in range(0, num_input_frames * offset, offset):
            ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

            merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
            merged.paste(im=ground_truth_image, box=(0, 0))
            merged.paste(im=ground_truth_image, box=(575 + 40, 0))
            merged.paste(im=ground_truth_image, box=(575 * 2 + 80, 0))
            merged = np.array(merged)
            cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)
            output_video.write(merged)

        for timestep in tqdm(range(offset * num_input_frames, num_frames, offset)):
            predicted_pointset_1 = real_model.predict(input_info_1)[0]
            predicted_pointset_2 = simulation_model.predict(input_info_2)[0]

            pixel_coordinates_1 = convert_to_pixel_coordinates(predicted_pointset_1[0], distance, height)
            cv2.fillPoly(img=background_image_1, pts=[pixel_coordinates_1], color=(47, 164, 193))

            pixel_coordinates_2 = convert_to_pixel_coordinates(predicted_pointset_2[0], distance, height)
            cv2.fillPoly(img=background_image_2, pts=[pixel_coordinates_2], color=(47, 164, 193))

            # concatenate with ground truth image for comparison
            ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

            merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
            merged.paste(im=ground_truth_image, box=(0, 0))
            merged.paste(im=Image.fromarray(cv2.resize(background_image_1, dsize=(575, 520))), box=(575 + 40, 0))
            merged.paste(im=Image.fromarray(cv2.resize(background_image_2, dsize=(575, 520))), box=(575 * 2 + 80, 0))
            merged = np.array(merged)

            cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)
            output_video.write(merged)
            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)
            del background_image_1, background_image_2
            background_image_1 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            background_image_2 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]

        output_video.release()
        cv2.destroyAllWindows()


def compare_baseline_ours_rendered(real_model_1, real_model_2, real_model_3,
                                   num_input_frames, save_path, offset, fps,
                                   data_type_1, data_type_2, data_type_3, output_video):

    global_losses_1 = np.array([0.0] * COMPARE_LENGTH)
    global_losses_2 = np.array([0.0] * COMPARE_LENGTH)
    global_losses_3 = np.array([0.0] * COMPARE_LENGTH)

    global_area_losses_1 = np.array([0.0] * COMPARE_LENGTH)
    global_area_losses_2 = np.array([0.0] * COMPARE_LENGTH)
    global_area_losses_3 = np.array([0.0] * COMPARE_LENGTH)

    chamfer_graph_save_path = create_directory(os.path.join(save_path, 'Loss Graph', 'Chamfer'))
    area_loss_graph_save_path = create_directory(os.path.join(save_path, 'Loss Graph', 'Area Loss'))

    for case in REAL_WORLD_VAL_CASES + REAL_WORLD_TEST_CASES:

        print(f'=========== Testing Case #{case} =============')

        timesteps = []
        losses_1 = []
        losses_2 = []
        losses_3 = []

        area_losses_1 = []
        area_losses_2 = []
        area_losses_3 = []

        ground_truth_base_path = os.path.join(REAL_DATA_PATH, '04_critical_frames_subtracted_cropped', f'case_{case}')
        first_frame_number, num_frames = find_case_info(ground_truth_base_path)
        ground_truth_pointset = get_real_world_ground_truth_pointset(case, data_type_1)

        input_info_1 = get_real_world_input_pointset(case, num_input_frames, offset, data_type_1)
        input_info_2 = get_real_world_input_pointset(case, num_input_frames, offset, data_type_2)
        input_info_3 = get_real_world_input_pointset(case, num_input_frames, offset, data_type_3)

        if output_video:
            distance, height = get_final_video_size(case)
            video_size = (575 * 4 + 120, 520)
            # video_size = ((228 + CROP_SIZE) * 2 + 50, 1040)
            # (1150 * 2 + 50, 1040)
            output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case}.MP4'), CODEC, fps, video_size)

            frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case}'))
            background_image_1 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            background_image_2 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            background_image_3 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]

            ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
            ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

            merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
            merged.paste(im=ground_truth_image, box=(0, 0))
            merged.paste(im=ground_truth_image, box=(575 + 40, 0))
            merged.paste(im=ground_truth_image, box=(575 * 2 + 80, 0))
            merged.paste(im=ground_truth_image, box=(575 * 3 + 120, 0))
            merged = np.array(merged)
            cv2.imwrite(os.path.join(frames_savepath, f'timestep_{0}.jpg'), merged)

            for i in range(20):
                output_video.write(merged)

            for timestep in range(0, num_input_frames * offset, offset):
                ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
                ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

                merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
                merged.paste(im=ground_truth_image, box=(0, 0))
                merged.paste(im=ground_truth_image, box=(575 + 40, 0))
                merged.paste(im=ground_truth_image, box=(575 * 2 + 80, 0))
                merged.paste(im=ground_truth_image, box=(575 * 3 + 120, 0))
                merged = np.array(merged)
                cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)
                output_video.write(merged)

        for timestep in tqdm(range(offset * num_input_frames, num_frames, offset)):
            predicted_pointset_1 = real_model_1.predict(input_info_1)[0]
            predicted_pointset_2 = real_model_2.predict(input_info_2)[0]
            predicted_pointset_3 = real_model_3.predict(input_info_3)[0]

            area_loss_1 = get_area_loss(np.array(ground_truth_pointset[timestep], dtype='float32'), predicted_pointset_1[0])
            area_loss_2 = get_area_loss(np.array(ground_truth_pointset[timestep], dtype='float32'), predicted_pointset_2[0])
            area_loss_3 = get_area_loss(np.array(ground_truth_pointset[timestep], dtype='float32'), predicted_pointset_3[0])

            loss_1 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_1)
            loss_2 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_2)
            loss_3 = get_cd_loss_func(np.array([ground_truth_pointset[timestep]], dtype='float32'), predicted_pointset_3)

            timesteps.append(timestep)
            losses_1.append(loss_1)
            losses_2.append(loss_2)
            losses_3.append(loss_3)

            area_losses_1.append(area_loss_1)
            area_losses_2.append(area_loss_2)
            area_losses_3.append(area_loss_3)

            if output_video:
                pixel_coordinates_1 = convert_to_pixel_coordinates(predicted_pointset_1[0], distance, height)
                cv2.fillPoly(img=background_image_1, pts=[pixel_coordinates_1], color=(47, 164, 193))

                pixel_coordinates_2 = convert_to_pixel_coordinates(predicted_pointset_2[0], distance, height)
                cv2.fillPoly(img=background_image_2, pts=[pixel_coordinates_2], color=(47, 164, 193))

                pixel_coordinates_3 = convert_to_pixel_coordinates(predicted_pointset_3[0], distance, height)
                cv2.fillPoly(img=background_image_3, pts=[pixel_coordinates_3], color=(47, 164, 193))

                # concatenate with ground truth image for comparison
                ground_truth_image = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', f'timestep_{first_frame_number + timestep}.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
                ground_truth_image = Image.fromarray(cv2.resize(ground_truth_image, dsize=(575, 520)))

                merged = Image.new(mode="RGB", size=video_size, color=(255, 255, 255))
                merged.paste(im=ground_truth_image, box=(0, 0))
                merged.paste(im=Image.fromarray(cv2.resize(background_image_1, dsize=(575, 520))), box=(575 + 40, 0))
                merged.paste(im=Image.fromarray(cv2.resize(background_image_2, dsize=(575, 520))), box=(575 * 2 + 80, 0))
                merged.paste(im=Image.fromarray(cv2.resize(background_image_3, dsize=(575, 520))), box=(575 * 3 + 120, 0))
                merged = np.array(merged)

                cv2.imwrite(os.path.join(frames_savepath, f'timestep_{timestep}.jpg'), merged)
                output_video.write(merged)

                del background_image_1, background_image_2, background_image_3

                background_image_1 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
                background_image_2 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]
                background_image_3 = cv2.imread(os.path.join(REAL_DATA_PATH, '02_critical_frames', f'case_{case}', 'timestep_0.jpg'), cv2.COLOR_BGR2RGB)[height - CROP_SIZE:height - CROP_SIZE + 1040, (1920 - distance - 228):(1920 - distance + CROP_SIZE)]

            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)
            input_info_3 = update_input_pointset(input_info_3, predicted_pointset_3)

        if output_video:
            output_video.release()
            cv2.destroyAllWindows()

        # Loss Graph (Chamfer Distance)
        plt.plot(timesteps, losses_1, label=f'{data_type_1}')
        plt.plot(timesteps, losses_2, label=f'{data_type_2}')
        plt.plot(timesteps, losses_3, label=f'{data_type_3}')

        plt.xlabel('Timestep')
        plt.ylabel('Chamfer Distance')
        plt.legend()
        plt.savefig(os.path.join(chamfer_graph_save_path, 'Test Case {}.png'.format(case)), dpi=600)
        plt.clf()

        plt.plot(timesteps, area_losses_1, label=f'{data_type_1}')
        plt.plot(timesteps, area_losses_2, label=f'{data_type_2}')
        plt.plot(timesteps, area_losses_3, label=f'{data_type_3}')

        # Loss Graph (Area Loss)
        plt.xlabel('Timestep')
        plt.ylabel('Area Loss')
        plt.legend()
        plt.savefig(os.path.join(area_loss_graph_save_path, 'Test Case {}.png'.format(case)), dpi=600)
        plt.clf()

        global_losses_1 += np.array(losses_1[:COMPARE_LENGTH])
        global_losses_2 += np.array(losses_2[:COMPARE_LENGTH])
        global_losses_3 += np.array(losses_3[:COMPARE_LENGTH])

        global_area_losses_1 += np.array(area_losses_1[:COMPARE_LENGTH])
        global_area_losses_2 += np.array(area_losses_2[:COMPARE_LENGTH])
        global_area_losses_3 += np.array(area_losses_3[:COMPARE_LENGTH])

    # Average Loss Graph - First COMPARE_LENGTH Frames
    global_losses_1 /= len(REAL_WORLD_TEST_CASES)
    global_losses_2 /= len(REAL_WORLD_TEST_CASES)
    global_losses_3 /= len(REAL_WORLD_TEST_CASES)

    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_1.tolist(), label=f'{data_type_1}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_2.tolist(), label=f'{data_type_2}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_3.tolist(), label=f'{data_type_3}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Chamfer Distance')
    plt.legend()
    plt.savefig(os.path.join(chamfer_graph_save_path, '..', 'Average.png'), dpi=600)
    plt.clf()

    # Average Loss Graph - First COMPARE_LENGTH Frames without Sorted
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_1.tolist(), label=f'{data_type_1}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_losses_2.tolist(), label=f'{data_type_2}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Chamfer Distance')
    plt.legend()
    plt.savefig(os.path.join(chamfer_graph_save_path, '..', 'Average (Unordered VS Ordered).png'), dpi=600)
    plt.clf()

    # Average Loss Graph - First 30 Frames
    plt.plot(timesteps[:30], global_losses_1.tolist()[:30], label=f'{data_type_1}')
    plt.plot(timesteps[:30], global_losses_2.tolist()[:30], label=f'{data_type_2}')
    plt.plot(timesteps[:30], global_losses_3.tolist()[:30], label=f'{data_type_3}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Chamfer Distance')
    plt.legend()
    plt.savefig(os.path.join(chamfer_graph_save_path, '..', 'Average First 30.png'), dpi=600)
    plt.clf()

    # Average Loss Graph - First 30 Frames Without Sorted
    plt.plot(timesteps[:30], global_losses_1.tolist()[:30], label=f'{data_type_1}')
    plt.plot(timesteps[:30], global_losses_2.tolist()[:30], label=f'{data_type_2}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Chamfer Distance')
    plt.legend()
    plt.savefig(os.path.join(chamfer_graph_save_path, '..', 'Average First 30 (Unordered VS Ordered).png'), dpi=600)
    plt.clf()

    # Average Area Loss Graph - First COMPARE_LENGTH Frames
    global_area_losses_1 /= len(REAL_WORLD_TEST_CASES)
    global_area_losses_2 /= len(REAL_WORLD_TEST_CASES)
    global_area_losses_3 /= len(REAL_WORLD_TEST_CASES)

    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_1.tolist(), label=f'{data_type_1}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_2.tolist(), label=f'{data_type_2}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_3.tolist(), label=f'{data_type_3}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(area_loss_graph_save_path, '..', 'Average.png'), dpi=600)
    plt.clf()

    # Average Area Loss Graph - First COMPARE_LENGTH Frames (Without Sorted)
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_1.tolist()[:COMPARE_LENGTH], label=f'{data_type_1}')
    plt.plot(timesteps[:COMPARE_LENGTH], global_area_losses_2.tolist()[:COMPARE_LENGTH], label=f'{data_type_2}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(area_loss_graph_save_path, '..', 'Average (Unordered VS Ordered).png'), dpi=600)
    plt.clf()

    # Average Area Loss Graph - First 30 Frames
    plt.plot(timesteps[:30], global_area_losses_1.tolist()[:30], label=f'{data_type_1}')
    plt.plot(timesteps[:30], global_area_losses_2.tolist()[:30], label=f'{data_type_2}')
    plt.plot(timesteps[:30], global_area_losses_3.tolist()[:30], label=f'{data_type_3}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(area_loss_graph_save_path, '..', 'Average First 30.png'), dpi=600)
    plt.clf()

    # Average Area Loss Graph - First 30 Frames (Without Sorted)
    plt.plot(timesteps[:30], global_area_losses_1.tolist()[:30], label=f'{data_type_1}')
    plt.plot(timesteps[:30], global_area_losses_2.tolist()[:30], label=f'{data_type_2}')

    plt.xlabel('Timestep')
    plt.ylabel('Average Area Loss')
    plt.legend()
    plt.savefig(os.path.join(area_loss_graph_save_path, '..', 'Average First 30 (Unordered VS Sorted).png'), dpi=600)
    plt.clf()
