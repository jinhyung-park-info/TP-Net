import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from common.Utils import create_directory, write_autoencoder_info
from loss import *
from prediction_models import *
import matplotlib.pyplot as plt
from common.Constants import RANDOM_SEED
import random
from data_generators import *
from autoencoder_models import *

# ========================= For Reproducibility ================================

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, default=100)
parser.add_argument('--batch_size', required=False, default=8)
parser.add_argument('--patience', required=False, default=10)
parser.add_argument('--lr', required=False, default=0.001)

parser.add_argument('--ver', required=False, default=2)
parser.add_argument('--model_type', required=False, default='encoder', choices=['encoder', 'decoder'])
parser.add_argument('--pointset_data_type', required=False, default='sorted', choices=['ordered', 'unordered', 'sorted'])
parser.add_argument('--loss_type', required=False, default='mse', choices=['mse', 'chamfer'])
parser.add_argument('--activation', required=False, default='relu', choices=['relu', 'leaky_relu'])
FLAGS = parser.parse_args()

BATCH_SIZE = int(FLAGS.batch_size)
EPOCHS = int(FLAGS.epochs)
PATIENCE = int(FLAGS.patience)
LEARNING_RATE = float(FLAGS.lr)
MODEL_VERSION = int(FLAGS.ver)
MODEL_TYPE = FLAGS.model_type
POINTSET_DATA_TYPE = FLAGS.pointset_data_type
LOSS_TYPE = FLAGS.loss_type
ACTIVATION = FLAGS.activation
BASE_PATH = create_directory(os.path.join('../result', MODEL_TYPE, f'version_{MODEL_VERSION}'))

print('========================= Preparing Data Generator =========================\n')

train_image_filepaths = np.load(os.path.join('../data', 'autoencoder', f'{POINTSET_DATA_TYPE}', 'train_img_paths.npy'))
train_pointset = np.load(os.path.join('../data', 'autoencoder', f'{POINTSET_DATA_TYPE}', f'train_pointset_{POINTSET_DATA_TYPE}.npy'))

val_image_filepaths = np.load(os.path.join('../data', 'autoencoder', f'{POINTSET_DATA_TYPE}', 'val_img_paths.npy'))
val_pointset = np.load(os.path.join('../data', 'autoencoder', f'{POINTSET_DATA_TYPE}', f'val_pointset_{POINTSET_DATA_TYPE}.npy'))


print('========================= Building Model =========================\n')

if MODEL_TYPE == 'encoder':
    train_batch_generator = EncoderDataGenerator(train_image_filepaths, BATCH_SIZE, labels=train_pointset)
    val_batch_generator = EncoderDataGenerator(val_image_filepaths, BATCH_SIZE, labels=val_pointset)

    model = build_encoder(ACTIVATION)
    if LOSS_TYPE == 'chamfer':
        model.compile(loss=get_cd_loss_func,
                      optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))
    else:
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

else:
    train_batch_generator = DecoderDataGenerator(train_pointset, BATCH_SIZE, image_filenames=train_image_filepaths)
    val_batch_generator = DecoderDataGenerator(val_pointset, BATCH_SIZE, image_filenames=val_image_filepaths)

    model = build_decoder()
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

model.summary()

print('========================= Training Model =========================\n')

write_autoencoder_info(path=os.path.join(BASE_PATH),
                       model=model,
                       model_type=MODEL_TYPE,
                       pointset_data_type=POINTSET_DATA_TYPE,
                       loss_function=LOSS_TYPE,
                       batch_size=BATCH_SIZE,
                       patience=PATIENCE,
                       lr=LEARNING_RATE,
                       activation=ACTIVATION)

tb = TensorBoard(log_dir=os.path.join(BASE_PATH, 'logs'))

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=2,
                   patience=PATIENCE)

cp = ModelCheckpoint(filepath=os.path.join(BASE_PATH, f'{MODEL_TYPE}.h5'),
                     save_weights_only=False,
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True,
                     save_freq='epoch',
                     verbose=1)

history = model.fit(train_batch_generator,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=val_batch_generator,
                    callbacks=[tb, es, cp])

model.save(os.path.join(BASE_PATH, f'{MODEL_TYPE}_final.h5'))

print('========================= Evaluating Model =========================\n')

# Loss in Text File
loss_file = open(os.path.join(BASE_PATH, 'loss.txt'), 'w')
train_evaluation = model.evaluate(train_batch_generator, batch_size=BATCH_SIZE)
val_evaluation = model.evaluate(val_batch_generator, batch_size=BATCH_SIZE)

print(f'Train Loss: {str(train_evaluation)}')
print(f'Validation Loss: {str(val_evaluation)}')

loss_file.write(f'Train loss: {str(train_evaluation)}\n')
loss_file.write(f'Validation loss: {str(val_evaluation)}\n')

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_loss))

for i in range(len(epochs)):
    loss_file.write(f'Epoch {i+1} : {train_loss[i]},  {val_loss[i]}\n')
loss_file.close()


print('========================= Plotting Loss Graph =========================\n')

# Loss Graph
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, val_loss, label='val')
plt.legend()
plt.savefig(os.path.join(BASE_PATH, 'loss.png'), dpi=600)

# Last 100 Epoch Loss Graph
plt.clf()
plt.plot(epochs[-100:], train_loss[-100:], label='train')
plt.plot(epochs[-100:], val_loss[-100:], label='val')
plt.legend()
plt.savefig(os.path.join(BASE_PATH, 'loss_last_100.png'), dpi=600)
