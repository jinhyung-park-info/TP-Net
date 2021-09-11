import numpy as np
import os


def get_data(args):
    x_train = np.load(os.path.join(args.data_path, f'x_train_pred_{args.data_type}.npy'))
    y_train = np.load(os.path.join(args.data_path, f'y_train_pred_{args.data_type}.npy'))

    x_val = np.load(os.path.join(args.data_path, f'x_val_pred_{args.data_type}.npy'))
    y_val = np.load(os.path.join(args.data_path, f'y_val_pred_{args.data_type}.npy'))

    y_train = [y_train[i] for i in range(args.num_output)]
    y_val = [y_val[i] for i in range(args.num_output)]

    args.n_train_data = int(x_train.shape[0])
    args.n_val_data = int(x_val.shape[0])

    return x_train, y_train, x_val, y_val
