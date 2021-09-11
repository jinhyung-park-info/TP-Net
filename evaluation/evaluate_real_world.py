import argparse
from common.Constants import REAL_DATA_EVAL_CASES
from tqdm import tqdm
import os
from evaluation.simulation_test_utils import load_pred_model, generate_rollout_error_graph
from evaluation.real_world_test_utils import get_real_world_input_pointset, get_error_for_real_data
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=False, default='tp_net')
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--num_input', required=False, default=4)
    parser.add_argument('--length', required=False, default=160)
    parser.add_argument('--init_data_type', required=False, default='unordered', choices=['unordered', 'sorted_x', 'sorted_y'])
    args = parser.parse_args()

    result_base_path = os.path.join('..', 'result', args.model_type, f'{args.model_type}-{args.num_input}', f'seed_{args.seed}')
    prediction_model_path = os.path.join(result_base_path, f'{args.model_type}_best.h5')

    test_cases = REAL_DATA_EVAL_CASES                  # 40 Real World Cases

    # 1. Generate Prediction File
    prediction_model = load_pred_model(prediction_model_path)
    predictions_merged = []

    print(f'============== Evaluating on Real World Dataset ==============')

    for i, test_case in tqdm(list(enumerate(test_cases))):

        prediction_per_test_case = []
        input_info = get_real_world_input_pointset(test_case, args.num_input, args.init_data_type)

        for _ in range(0, args.length, 8):   # model outputs 8 frames per prediction
            predicted_pointset = prediction_model.predict(input_info)
            for j in range(8):
                prediction_per_test_case.append(predicted_pointset[j][0])

            if args.num_input <= 8:
                input_info = np.array(predicted_pointset[-args.num_input:]).reshape((1, args.num_input, 30, 2))
            else:
                input_info = np.array(input_info[:, -(args.num_input - args.num_output):])
                predicted_pointset = np.array(predicted_pointset).reshape((1, args.num_output, 30, 2))
                input_info = np.concatenate((input_info, predicted_pointset), axis=1)

        predictions_merged.append(prediction_per_test_case)

    np.save(os.path.join(result_base_path, f'real_softbody_predictions_{args.init_data_type}.npy'), np.array(predictions_merged))

    # 2. Compute Error
    for error_type in ['Position', 'Shape']:

        print(f'============== Computing {error_type} Error ==============')

        errors = get_error_for_real_data(model_type=args.model_type, seed=args.seed, num_input=args.num_input,
                                         error_type=error_type, data_type=args.init_data_type)

        np.save(os.path.join(result_base_path, f'{args.model_type}-{args.num_input}-{error_type}-real-{args.init_data_type}-errors.npy'), np.array(errors))
        errors_average = np.mean(errors, axis=0)
        np.save(os.path.join(result_base_path, f'{args.model_type}-{args.num_input}-{error_type}-real-{args.init_data_type}-errors-average.npy'), errors_average)
        generate_rollout_error_graph(args, result_base_path, errors_average, error_type, env='real_world')
