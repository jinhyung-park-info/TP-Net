import numpy as np
import json
from common.Constants import *
import random
from simulator.WallFactory import *


# Simulation Related
class ContactListener(b2ContactListener):

    def __init__(self, world):
        b2ContactListener.__init__(self)
        self.world = world

    def BeginContact(self, contact):
        print("BeginContact")

    def EndContact(self, contact):
        print("EndContact")


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


def denormalize_pointset(pointset):
    denormalized = []

    for j in range(NUM_PARTICLES):
        denormalized_x = pointset[j][0] * (SAT_MAX - SAT_MIN) + SAT_MIN
        denormalized_y = pointset[j][1] * (SAT_MAX - SAT_MIN) + SAT_MIN
        assert SAT_MIN <= denormalized_x <= SAT_MAX
        assert SAT_MIN <= denormalized_y <= SAT_MAX
        denormalized.append([denormalized_x, denormalized_y])

    assert len(denormalized) == NUM_PARTICLES
    return denormalized


def shuffle_pointset(pointset):
    return random.sample(pointset, len(pointset))


def sort_pointset(pointset):
    return sorted(pointset, key=lambda coord: (-coord[1], coord[0]))


def is_within_collision_range(pointset):
    max_val = max(pointset)
    min_val = min(pointset)
    if max_val >= SAT_MAX - COLLISION_RANGE_BOUND:
        return True
    if min_val <= SAT_MIN + COLLISION_RANGE_BOUND:
        return True
    else:
        return False


def write_model_info(path,
                     model,
                     loss_function,
                     shape_weight,
                     model_type,
                     use_transform,
                     data_type,
                     offset,
                     train_input,
                     train_output,
                     val_input,
                     val_output,
                     batch_size,
                     patience):

    file = open(os.path.join(path, 'README.txt'), 'w')

    file.write('=============== Model Info =============\n\n')

    file.write(f'Model Type                : {model_type}\n')
    file.write(f'Uses Transform Net        : {use_transform}\n')
    file.write(f'Loss Type                 : {loss_function}\n')
    file.write(f'Shape Loss Weight         : {shape_weight}\n')
    file.write(f'Data Type                 : {data_type}\n')
    file.write(f'Data Offset               : {offset}\n')
    file.write(f'Train Data Input Shape    : {train_input}\n')
    file.write(f'Train Data Output Shape   : {train_output}\n')
    file.write(f'Val Data Input Shape      : {val_input}\n')
    file.write(f'Val Data Output Shape     : {val_output}\n')
    file.write(f'Batch Size                : {batch_size}\n')
    file.write(f'Patience                  : {patience}\n')
    file.write(f'Random Seed               : {RANDOM_SEED}\n\n\n')

    file.write('=============== Single Frame Prediction Model Summary =============\n\n')
    model.get_layer('functional_3').summary(print_fn=lambda x: file.write(x + '\n\n\n'))
    file.close()


def write_real_model_info(path,
                          model,
                          simulation_base_model_ver,
                          retrain_scope,
                          model_type,
                          data_type,
                          loss_type,
                          shape_weight,
                          offset,
                          train_input,
                          train_output,
                          val_input,
                          val_output,
                          batch_size,
                          patience,
                          lr):

    file = open(os.path.join(path, 'README.txt'), 'w')

    file.write('=============== Model Info =============\n\n')

    file.write(f'Simulation Model Ver      : {simulation_base_model_ver}\n')
    file.write(f'Retrain Scope             : {retrain_scope}\n')
    file.write(f'Model Type                : {model_type}\n')
    file.write(f'Input Data Type           : {data_type}\n')
    file.write(f'Loss Type                 : {loss_type}\n')
    file.write(f'Shape Weight              : {shape_weight}\n')
    file.write(f'Data Offset               : {offset}\n')
    file.write(f'Train Data Input Shape    : {train_input}\n')
    file.write(f'Train Data Output Shape   : {train_output}\n')
    file.write(f'Val Data Input Shape      : {val_input}\n')
    file.write(f'Val Data Output Shape     : {val_output}\n')
    file.write(f'Batch Size                : {batch_size}\n')
    file.write(f'Patience                  : {patience}\n')
    file.write(f'Learning Rate             : {lr}\n')
    file.write(f'Random Seed               : {RANDOM_SEED}\n\n\n')

    file.write('=============== Model Summary =============\n\n')
    model.summary(print_fn=lambda x: file.write(x + '\n\n\n'))
    file.close()


def write_autoencoder_info(path,
                           model,
                           model_type,
                           pointset_data_type,
                           loss_function,
                           batch_size,
                           patience,
                           lr,
                           activation):

    file = open(os.path.join(path, 'README.txt'), 'w')

    file.write('=============== Model Info =============\n\n')

    file.write(f'Model Type                : {model_type}\n')
    file.write(f'PointSet Data Type        : {pointset_data_type}\n')
    file.write(f'Loss Type                 : {loss_function}\n')
    file.write(f'Batch Size                : {batch_size}\n')
    file.write(f'Patience                  : {patience}\n')
    file.write(f'Learning Rate             : {lr}\n')
    file.write(f'Activation                : {activation}\n\n\n')

    file.write('=============== Model Summary =============\n\n')
    model.summary(print_fn=lambda x: file.write(x + '\n\n\n'))
    file.close()
