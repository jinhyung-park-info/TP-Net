from data.simulation.preprocess_code.preprocess_utils import *

if __name__ == '__main__':

    # ========================= For Reproducibility ================================
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Prediction Model
    OFFSET = 4
    NUM_PREDICTIONS = 1
    # print(find_min_max())

    generate_prediction_model_data(NUM_ANIMATIONS, NUM_PREDICTIONS, OFFSET)

    # Autoencoder
    # NUM_USE_ANIMATIONS = 6000
    # CASE_PER_ANIMATION = 25
    # generate_autoencoder_data(NUM_USE_ANIMATIONS, CASE_PER_ANIMATION)
