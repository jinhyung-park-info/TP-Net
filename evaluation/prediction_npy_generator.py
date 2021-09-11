import argparse
from common.Constants import EVAL_CASES, REAL_WORLD_EVAL_CASES
from tqdm import tqdm
import os
from evaluation.simulation_test_utils import load_pred_model, get_simulation_input_pointset
from evaluation.real_world_test_utils import get_real_world_input_pointset
import numpy as np


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=False, default='tp_net', choices=['global_pointnet', 'tp_net'])
    parser.add_argument('--num_input', required=False, default=5)
    parser.add_argument('--num_output', required=False, default=8)
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--start_timestep', required=False, default=0)
    parser.add_argument('--offset', required=False, default=2)
    parser.add_argument('--length', required=False, default=200)  # must be a multiple of num_output
    parser.add_argument('--test_data_type', required=False, default='ordered', choices=['ordered', 'unordered', 'sorted'])
    parser.add_argument('--env', required=False, default='real', choices=['simulation', 'real'])
    parser.add_argument('--num_test_animations', required=False, default=100)
    args = parser.parse_args()

    result_base_path = os.path.join('..', 'result', args.model_type, f'{args.model_type}-{args.num_input}', f'seed_{args.seed}')
    if args.model_type == 'tp_net':
        prediction_model_path = os.path.join(result_base_path, f'{args.model_type}_epoch_120.h5')
    else:
        prediction_model_path = os.path.join(result_base_path, f'{args.model_type}_model.h5')

    if args.env == 'simulation':
        test_cases = EVAL_CASES[:args.num_test_animations]
    else:
        test_cases = REAL_WORLD_EVAL_CASES

    prediction_model = load_pred_model(prediction_model_path)
    all_predictions = []

    for i, test_case in enumerate(test_cases):
        print(f'============== Testing Case #{i + 1} ==============')
        predictions = []
        if args.env == 'simulation':
            input_info = get_simulation_input_pointset(test_case, args.offset, args.test_data_type, args.num_input, args.start_timestep)
        else:
            input_info = get_real_world_input_pointset(test_case, args.num_input, args.offset, args.test_data_type)

        for _ in tqdm(range(0, args.length, args.num_output)):
            predicted_pointset = prediction_model.predict(input_info)
            for j in range(args.num_output):
                predictions.append(predicted_pointset[j][0])
            if args.num_input <= args.num_output:
                input_info = np.array(predicted_pointset[-args.num_input:]).reshape((1, args.num_input, 30, 2))
            else:
                input_info = np.array(input_info[:, -(args.num_input - args.num_output):])
                predicted_pointset = np.array(predicted_pointset).reshape((1, args.num_output, 30, 2))
                input_info = np.concatenate((input_info, predicted_pointset), axis=1)

        all_predictions.append(predictions)

    if args.env == 'simulation':
        if args.start_timestep == 0:
            np.save(os.path.join(result_base_path, f'softbody_predictions.npy'), np.array(all_predictions))
        else:
            np.save(os.path.join(result_base_path, f'softbody_predictions_{args.start_timestep}.npy'), np.array(all_predictions))
    else:
        if args.start_timestep == 0:
            if args.test_data_type != 'ordered':
                np.save(os.path.join(result_base_path, f'real_softbody_predictions_{args.test_data_type}.npy'), np.array(all_predictions))
            else:
                np.save(os.path.join(result_base_path, f'real_softbody_predictions.npy'), np.array(all_predictions))
        else:
            np.save(os.path.join(result_base_path, f'real_softbody_predictions_{args.start_timestep}.npy'), np.array(all_predictions))
