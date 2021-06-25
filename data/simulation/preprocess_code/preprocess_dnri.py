from data.simulation.preprocess_code.preprocess_dnri_utils import *

if __name__ == '__main__':

    # ========================= For Reproducibility ================================
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Prediction Model
    OFFSET = 4
    generate_prediction_model_data_for_dnri(NUM_ANIMATIONS, OFFSET, include_vel=True, only_eval_data=False)
