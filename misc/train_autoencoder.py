import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from autoencoder.data_generators import EncoderDataGenerator, DecoderDataGenerator
from common.Utils import create_directory
from prediction_models import *
import matplotlib.pyplot as plt
from common.Constants import RANDOM_SEED
from loss import *

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser(description='Image to PointSet Autoencoder')
parser.add_argument('--epochs', required=False, default=100)
parser.add_argument('--batch_size', required=False, default=8)
parser.add_argument('--patience', required=False, default=10)
parser.add_argument('--num_train', required=False, default=50000, help='number of training data to use <= 120000')
parser.add_argument('--num_val', required=False, default=5000, help='number of validation data to use <= 3000')
parser.add_argument('--lr', required=False, default=0.001, help='learning rate')
parser.add_argument('--model_type', required=False, default='decoder', help='Options: (1) encoder (2) decoder')
parser.add_argument('--ver', required=False, default=7, help='version number of the model')
FLAGS = parser.parse_args()

BATCH_SIZE = int(FLAGS.batch_size)
NUM_TRAIN = int(FLAGS.num_train)
NUM_VAL = int(FLAGS.num_val)
EPOCHS = int(FLAGS.epochs)
PATIENCE = int(FLAGS.patience)
LEARNING_RATE = float(FLAGS.lr)
MODEL_TYPE = FLAGS.model_type
MODEL_VERSION = int(FLAGS.ver)
BASE_PATH = create_directory(os.path.join('../result', MODEL_TYPE, f'version_{MODEL_VERSION}'))

tf.random.set_seed(RANDOM_SEED)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


print('========================= Preparing Data Generator =========================\n')

train_image_filepaths = np.load(os.path.join('../data', 'simulation', 'train_img_paths.npy'))[:NUM_TRAIN]
train_pointset = np.load(os.path.join('../data', 'simulation', 'train_pointset.npy'))[:NUM_TRAIN]

val_image_filepaths = np.load(os.path.join('../data', 'simulation', 'val_img_paths.npy'))[:NUM_VAL]
val_pointset = np.load(os.path.join('../data', 'simulation', 'val_pointset.npy'))[:NUM_VAL]


print('========================= Building Model =========================\n')

if MODEL_TYPE == 'encoder':
    train_batch_generator = EncoderDataGenerator(train_image_filepaths, BATCH_SIZE, labels=train_pointset)
    val_batch_generator = EncoderDataGenerator(val_image_filepaths, BATCH_SIZE, labels=val_pointset)

    model = build_encoder()
    model.compile(loss=chamfer_and_mae,
                  optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['mae'])

else:
    train_batch_generator = DecoderDataGenerator(train_pointset, BATCH_SIZE, image_filenames=train_image_filepaths)
    val_batch_generator = DecoderDataGenerator(val_pointset, BATCH_SIZE, image_filenames=val_image_filepaths)

    model = build_decoder_for_high_res_image()
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['mae'])

model.summary()


print('========================= Training Model =========================\n')

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


print('========================= Evaluating Model =========================\n')

# Loss in Text File
loss_file = open(os.path.join(BASE_PATH, 'loss.txt'), 'w')
train_evaluation = model.evaluate(train_batch_generator, batch_size=BATCH_SIZE)
val_evaluation = model.evaluate(val_batch_generator, batch_size=BATCH_SIZE)

print(f'Train Loss: {str(train_evaluation)}')
print(f'Validation Loss: {str(val_evaluation)}')

loss_file.write(f'Train loss: {str(train_evaluation)}\n')
loss_file.write(f'Validation loss: {str(val_evaluation)}\n\n\n')

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']
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

# Mae Graph
plt.clf()
plt.plot(epochs, train_mae, label='train')
plt.plot(epochs, val_mae, label='val')
plt.legend()
plt.savefig(os.path.join(BASE_PATH, 'mae.png'), dpi=600)

# Last 30 Epoch Loss Graph
plt.clf()
plt.plot(epochs[-30:], train_loss[-30:], label='train')
plt.plot(epochs[-30:], val_loss[-30:], label='val')
plt.legend()
plt.savefig(os.path.join(BASE_PATH, 'loss_last_30.png'), dpi=600)


# Last 30 Epoch Mae Graph
plt.clf()
plt.plot(epochs[-30:], train_mae[-30:], label='train')
plt.plot(epochs[-30:], val_mae[-30:], label='val')
plt.legend()
plt.savefig(os.path.join(BASE_PATH, 'mae_last_30.png'), dpi=600)
