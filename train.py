import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from common.Utils import create_directory
from loss import *
from prediction_models import *
import matplotlib.pyplot as plt
from common.Constants import RANDOM_SEED
import numpy as np
from common.Utils import write_model_info
import random

# ========================= For Reproducibility ================================

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', required=False, default=400)
parser.add_argument('--batch_size', required=False, default=128)
parser.add_argument('--patience', required=False, default=100)
parser.add_argument('--lr', required=False, default=0.001)

parser.add_argument('--model_type', required=False, default='global_pointnet', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--transform', required=False, default=1)
parser.add_argument('--ver', required=False, default=47)
parser.add_argument('--data_offset', required=False, default=4)
parser.add_argument('--data_type', required=False, default='unordered', choices=['ordered', 'unordered', 'sorted'])
parser.add_argument('--num_predictions', required=False, default=8)
parser.add_argument('--loss_type', required=False, default='chamfer')
parser.add_argument('--shape_weight', required=False, default=0)

FLAGS = parser.parse_args()

BATCH_SIZE = int(FLAGS.batch_size)
EPOCHS = int(FLAGS.epochs)
PATIENCE = int(FLAGS.patience)
LEARNING_RATE = float(FLAGS.lr)
MODEL_TYPE = FLAGS.model_type
USE_TRANSFORM = bool(int(FLAGS.transform))
MODEL_VERSION = int(FLAGS.ver)
DATA_OFFSET = int(FLAGS.data_offset)
DATA_TYPE = FLAGS.data_type
NUM_PREDICTIONS = int(FLAGS.num_predictions)
LOSS_TYPE = FLAGS.loss_type
SHAPE_WEIGHT = float(FLAGS.shape_weight)
BASE_PATH = create_directory(os.path.join('./result', MODEL_TYPE, f'version_{MODEL_VERSION}'))

print('========================= Loading Data =========================\n')

x_train = np.load(f'./data/simulation/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT_FRAMES}_output_{NUM_PREDICTIONS}/x_train_pred_{DATA_TYPE}.npy')
y_train = np.load(f'./data/simulation/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT_FRAMES}_output_{NUM_PREDICTIONS}/y_train_pred_{DATA_TYPE}.npy')

x_val = np.load(f'./data/simulation/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT_FRAMES}_output_{NUM_PREDICTIONS}/x_val_pred_{DATA_TYPE}.npy')
y_val = np.load(f'./data/simulation/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT_FRAMES}_output_{NUM_PREDICTIONS}/y_val_pred_{DATA_TYPE}.npy')

y_train = [y_train[i] for i in range(NUM_PREDICTIONS)]
y_val = [y_val[i] for i in range(NUM_PREDICTIONS)]

print('========================= Building Model =========================\n')

if MODEL_TYPE == 'global_pointnet':
    model = build_recursive_prediction_model(global_pointnet_lstm(num_global_features=128, use_transform_net=USE_TRANSFORM, bn=False))

elif MODEL_TYPE == 'local_pointnet':
    model = build_recursive_prediction_model(pointnet_lstm())

elif MODEL_TYPE == 'lstm':
    model = build_recursive_prediction_model(simple_lstm())

else:
    model = None
    exit(0)

if LOSS_TYPE == 'chamfer':
    first_loss = get_cd_loss_func_for_first
    base_loss = get_cd_loss_func
elif LOSS_TYPE == 'chamfer_and_mae':
    first_loss = chamfer_and_mae_for_first
    base_loss = chamfer_and_mae
elif LOSS_TYPE == 'chamfer_and_mse':
    first_loss = chamfer_and_mse_for_first
    base_loss = chamfer_and_mse
elif LOSS_TYPE == 'mse':
    first_loss = mse_for_first
    base_loss = mse_base
else:
    first_loss = chamfer_and_shape_for_first
    base_loss = chamfer_and_shape


model.compile(loss={'tf_op_layer_output1': first_loss,
                    'tf_op_layer_output2': base_loss,
                    'tf_op_layer_output3': base_loss,
                    'tf_op_layer_output4': base_loss,
                    'tf_op_layer_output5': base_loss,
                    'tf_op_layer_output6': base_loss,
                    'tf_op_layer_output7': base_loss,
                    'tf_op_layer_output8': base_loss},
              optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, clipvalue=5))
model.summary()

print('========================= Training Model =========================\n')

write_model_info(path=os.path.join(BASE_PATH),
                 model=None,
                 model_type=MODEL_TYPE,
                 use_transform=str(USE_TRANSFORM),
                 loss_function=LOSS_TYPE,
                 shape_weight=SHAPE_WEIGHT,
                 data_type=DATA_TYPE,
                 offset=DATA_OFFSET,
                 train_input=x_train.shape,
                 train_output=(NUM_PREDICTIONS, ) + y_train[0].shape,
                 val_input=x_val.shape,
                 val_output=(NUM_PREDICTIONS, ) + y_val[0].shape,
                 batch_size=BATCH_SIZE,
                 patience=PATIENCE)

tb = TensorBoard(log_dir=os.path.join(BASE_PATH, 'logs'))

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=2,
                   patience=PATIENCE)

cp = ModelCheckpoint(filepath=os.path.join(BASE_PATH, f'{MODEL_TYPE}_model.h5'),
                     save_weights_only=False,
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True,
                     save_freq='epoch',
                     verbose=1)

history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[tb, es, cp])

model.save(os.path.join(BASE_PATH, f'{MODEL_TYPE}_model_final.h5'))

print('========================= Evaluating Model =========================\n')

# Loss in Text File
loss_file = open(os.path.join(BASE_PATH, 'loss.txt'), 'w')
train_evaluation = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
val_evaluation = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)

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
