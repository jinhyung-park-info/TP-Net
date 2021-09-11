import numpy as np
import json
from common.Constants import *
import random


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    except OSError:
        print('Error: Creating directory. ' + directory)
        exit(1)


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    with open(fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

    return f


def load_json(fname):
    with open(fname, encoding="utf-8-sig") as f:
        json_obj = json.load(f)

    return json_obj, f


def get_nested_pointset(pointset):
    nested = []
    for i in range(NUM_PARTICLES):
        nested.append([pointset[2*i], pointset[2*i + 1]])
    return nested


def normalize_pointset(pointset):
    normalized = []

    for j in range(2 * NUM_PARTICLES):
        normalized_element = (pointset[j] - SAT_MIN) / (SAT_MAX - SAT_MIN)
        assert 0 <= normalized_element <= 1.0
        normalized.append(normalized_element)

    assert len(normalized) == 2 * NUM_PARTICLES
    nested_normalized = []
    for i in range(NUM_PARTICLES):
        nested_normalized.append([normalized[2*i], normalized[2*i + 1]])

    return nested_normalized


def normalize_nested_pointset(pointset):
    normalized = []

    for j in range(NUM_PARTICLES):
        normalized_x = (pointset[j][0] - SAT_MIN) / (SAT_MAX - SAT_MIN)
        normalized_y = (pointset[j][1] - SAT_MIN) / (SAT_MAX - SAT_MIN)
        if normalized_x < 0:
            normalized_x = 0.0

        elif normalized_x > 1.0:
            normalized_x = 1.0

        if normalized_y < 0:
            normalized_y = 0.0

        elif normalized_y > 1.0:
            normalized_y = 1.0

        normalized.append([normalized_x, normalized_y])

    assert len(normalized) == NUM_PARTICLES
    return normalized


def denormalize_pointset(pointset):
    denormalized = []

    for j in range(NUM_PARTICLES):
        denormalized_x = pointset[j][0] * (SAT_MAX - SAT_MIN) + SAT_MIN
        denormalized_y = pointset[j][1] * (SAT_MAX - SAT_MIN) + SAT_MIN

        if SAT_MIN > denormalized_x:
            denormalized_x = SAT_MIN
        elif SAT_MAX < denormalized_x:
            denormalized_x = SAT_MAX

        if SAT_MIN > denormalized_y:
            denormalized_y = SAT_MIN
        elif SAT_MAX < denormalized_x:
            denormalized_y = SAT_MAX

        denormalized.append([denormalized_x, denormalized_y])

    assert len(denormalized) == NUM_PARTICLES
    return denormalized


def denormalize_dnri_pointset(pointset):
    denormalized = []

    for j in range(NUM_PARTICLES):
        denormalized_x = (pointset[j][0] + 1) * (SAT_MAX - SAT_MIN) / 2 + SAT_MIN
        denormalized_y = (pointset[j][1] + 1) * (SAT_MAX - SAT_MIN) / 2 + SAT_MIN
        denormalized.append([denormalized_x, denormalized_y])

    assert len(denormalized) == NUM_PARTICLES
    return denormalized


def shuffle_pointset(pointset):
    return random.sample(pointset, len(pointset))


def sort_pointset_by_descending_y(pointset):
    return sorted(pointset, key=lambda coord: (-coord[1], coord[0]))


def sort_pointset_by_ascending_x(pointset):
    return sorted(pointset, key=lambda coord: (coord[0], -coord[1]))


def center_transform(pointset):
    sum_info = np.sum(pointset[0], axis=0)
    center_x = sum_info[0] / NUM_PARTICLES
    center_y = sum_info[1] / NUM_PARTICLES
    for i in range(NUM_PARTICLES):
        pointset[0][i][0] -= center_x
        pointset[0][i][1] -= center_y
    return pointset


def is_within_collision_range(pointset):
    max_val = max(pointset)
    min_val = min(pointset)
    if max_val >= SAT_MAX - COLLISION_RANGE_BOUND:
        return True
    if min_val <= SAT_MIN + COLLISION_RANGE_BOUND:
        return True
    else:
        return False

