import argparse
import numpy as np
from evaluation.real_world_test_utils import get_real_error
from common.Utils import create_directory
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=False, default='tp_net', choices=['dpi_recursive', 'global_pointnet', 'dpi', 'dnri', 'static_nri', 'dynamic_nri', 'graphrnn'])
    parser.add_argument('--seed', required=False, default=3)
    parser.add_argument('--num_samples', required=False, default=15)
    parser.add_argument('--num_input', required=False, default=4)
    parser.add_argument('--start_timestep', required=False, default=0)
    parser.add_argument('--test_length', required=False, default=80)
    parser.add_argument('--offset', required=False, default=2)
    parser.add_argument('--error_type', required=False, default='Shape', choices=['Position', 'Shape'])
    parser.add_argument('--data_type', required=False, default='unordered')
    args = parser.parse_args()

    if args.model_type[-4:] == '_nri':
        savepath = create_directory(f'../result/nri/nri-{args.num_input}/seed_{args.seed}')
    else:
        savepath = create_directory(f'../result/{args.model_type}/{args.model_type}-{args.num_input}/seed_{args.seed}')

    print(f'=========================== Computing {args.error_type} Errors ===========================')
    errors = get_real_error(model_type=args.model_type, seed=args.seed, num_input=args.num_input,
                            error_type=args.error_type, data_type=args.data_type, offset=args.offset, start_timestep=args.start_timestep,
                            test_length=args.test_length, num_samples=args.num_samples)

    if args.model_type == 'graphrnn':
        np.save(os.path.join(savepath, f'{args.model_type}-{args.num_input}-{args.start_timestep}-{args.error_type}-real-{args.data_type}-errors-num_samples-{args.num_samples}.npy'), np.array(errors))
        errors_average = np.mean(errors, axis=0)
        np.save(os.path.join(savepath, f'{args.model_type}-{args.num_input}-{args.start_timestep}-{args.error_type}-real-{args.data_type}-errors-average-num_samples-{args.num_samples}.npy'), errors_average)

    else:
        np.save(os.path.join(savepath, f'{args.model_type}-{args.num_input}-{args.start_timestep}-{args.error_type}-real-{args.data_type}-errors.npy'), np.array(errors))
        errors_average = np.mean(errors, axis=0)
        np.save(os.path.join(savepath, f'{args.model_type}-{args.num_input}-{args.start_timestep}-{args.error_type}-real-{args.data_type}-errors-average.npy'), errors_average)
