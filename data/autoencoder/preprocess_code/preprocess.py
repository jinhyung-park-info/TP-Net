from data.autoencoder.preprocess_code.preprocess_utils import *
import os

NUM_USE_ANIMATIONS = 6000
CASE_PER_ANIMATION = 15
MODE = "sorted"

if __name__ == '__main__':
    # ========================= For Reproducibility ================================
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    generate_autoencoder_data(NUM_USE_ANIMATIONS, CASE_PER_ANIMATION, MODE)
