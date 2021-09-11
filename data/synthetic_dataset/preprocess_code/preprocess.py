from data.synthetic_dataset.preprocess_code.preprocess_utils import *

if __name__ == '__main__':

    # For Reproducibility
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    random.seed(1)

    # print(find_min_max())

    prediction_horizon = 8
    for n_input in [3, 4, 5]:
        generate_prediction_model_data(NUM_ANIMATIONS, n_input, prediction_horizon, SIM_DATA_OFFSET)
