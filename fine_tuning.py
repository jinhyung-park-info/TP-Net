import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from common.Utils import create_directory
from loss import *
from prediction_models import *
import matplotlib.pyplot as plt
from common.Constants import RANDOM_SEED
import numpy as np
from common.Utils import load_json, write_real_model_info
import random
from tensorflow.keras.models import load_model

# ========================= For Reproducibility ================================

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser('Fine Tuning Model')
parser.add_argument('--epochs', required=False, default=200)
parser.add_argument('--batch_size', required=False, default=32)
parser.add_argument('--patience', required=False, default=40)
parser.add_argument('--lr', required=False, default=0.0001, help='Must be 1/10 of original lr')

# Loading Model Related
parser.add_argument('--model_type', required=False, default='lstm', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--retrain_ver', required=False, default=24)
parser.add_argument('--data_type', required=False, default='sorted', choices=['unordered', 'ordered', 'sorted'])
parser.add_argument('--data_offset', required=False, default=2)
parser.add_argument('--num_input_frames', required=False, default=3)
parser.add_argument('--num_output_frames', required=False, default=8)
parser.add_argument('--loss_type', required=False, default='chamfer_and_mae', choices=['chamfer', 'mse', 'chamfer_and_shape', 'chamfer_and_mae'])
parser.add_argument('--shape_weight', required=False, default=0)

# Output Model Related
parser.add_argument('--ver', required=False, default=1)
parser.add_argument('--retrain_scope', required=False, default='full', choices=['full', 'lstm', 'lstm_last'])

FLAGS = parser.parse_args()

BATCH_SIZE = int(FLAGS.batch_size)
EPOCHS = int(FLAGS.epochs)
PATIENCE = int(FLAGS.patience)
LEARNING_RATE = float(FLAGS.lr)

MODEL_TYPE = FLAGS.model_type
RETRAIN_VERSION = int(FLAGS.retrain_ver)
DATA_TYPE = FLAGS.data_type
DATA_OFFSET = int(FLAGS.data_offset)
NUM_OUTPUT = int(FLAGS.num_output_frames)
NUM_INPUT = int(FLAGS.num_input_frames)
MODEL_VERSION = int(FLAGS.ver)
LOSS_TYPE = FLAGS.loss_type
SHAPE_WEIGHT = float(FLAGS.shape_weight)
RETRAIN_SCOPE = FLAGS.retrain_scope

BASE_PATH = create_directory(os.path.join('./result', MODEL_TYPE, f'version_{RETRAIN_VERSION}', 'fine_tuning', f'version_{MODEL_VERSION}'))

print('========================= Preparing Model =========================\n')

model = load_model(filepath=os.path.join('result', f'{MODEL_TYPE}', f'version_{RETRAIN_VERSION}', f'{MODEL_TYPE}_model.h5'), compile=False)
model.summary()

if RETRAIN_SCOPE == 'lstm':
    for layer in model.layers:
        if layer != model.get_layer('functional_3'):
            layer.trainable = False

    single_frame_prediction_model = model.get_layer('functional_3')
    for i, layer in enumerate(single_frame_prediction_model.layers):
        if i < 10:
            layer.trainable = False
        else:
            layer.trainable = True

elif RETRAIN_SCOPE == 'lstm_last':
    for layer in model.layers:
        if layer != model.get_layer('functional_3'):
            layer.trainable = False

    single_frame_prediction_model = model.get_layer('functional_3')
    for i, layer in enumerate(single_frame_prediction_model.layers):
        if i < 14:
            layer.trainable = False
        else:
            layer.trainable = True

else:
    for layer in model.layers:
        layer.trainable = True
    if MODEL_TYPE == 'global_pointnet':
        single_frame_prediction_model = model.get_layer('functional_3')
    else:
        single_frame_prediction_model = model.get_layer('functional_1')


if LOSS_TYPE == 'chamfer':
    first_loss = get_cd_loss_func_for_first
    base_loss = get_cd_loss_func

elif LOSS_TYPE == 'mse':
    first_loss = mse_for_first
    base_loss = 'mse'

elif LOSS_TYPE == 'chamfer_and_mae':
    first_loss = chamfer_and_mae_for_first
    base_loss = chamfer_and_mae
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
              optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))


single_frame_prediction_model.summary()

print('========================= Loading Data =========================\n')

x_train, ptr = load_json(f'data/real_world/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/x_train_pred_{DATA_TYPE}.json')
ptr.close()
y_train, ptr = load_json(f'data/real_world/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/y_train_pred_{DATA_TYPE}.json')
ptr.close()

x_val, ptr = load_json(f'data/real_world/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/x_val_pred_{DATA_TYPE}.json')
ptr.close()
y_val, ptr = load_json(f'data/real_world/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/y_val_pred_{DATA_TYPE}.json')
ptr.close()

y_train_0 = np.array(y_train[0])
y_train_1 = np.array(y_train[1])
y_train_2 = np.array(y_train[2])
y_train_3 = np.array(y_train[3])
y_train_4 = np.array(y_train[4])
y_train_5 = np.array(y_train[5])
y_train_6 = np.array(y_train[6])
y_train_7 = np.array(y_train[7])
y_train = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7]

y_val_0 = np.array(y_val[0])
y_val_1 = np.array(y_val[1])
y_val_2 = np.array(y_val[2])
y_val_3 = np.array(y_val[3])
y_val_4 = np.array(y_val[4])
y_val_5 = np.array(y_val[5])
y_val_6 = np.array(y_val[6])
y_val_7 = np.array(y_val[7])
y_val = [y_val_0, y_val_1, y_val_2, y_val_3, y_val_4, y_val_5, y_val_6, y_val_7]

x_train = np.array(x_train)
x_val = np.array(x_val)

print('========================= Training Model =========================\n')

write_real_model_info(path=os.path.join(BASE_PATH),
                      model=single_frame_prediction_model,
                      simulation_base_model_ver=RETRAIN_VERSION,
                      retrain_scope=RETRAIN_SCOPE,
                      model_type=MODEL_TYPE,
                      data_type=DATA_TYPE,
                      loss_type=MODEL_TYPE,
                      shape_weight=SHAPE_WEIGHT,
                      offset=DATA_OFFSET,
                      train_input=x_train.shape,
                      train_output=(NUM_OUTPUT, ) + y_train_0.shape,
                      val_input=x_val.shape,
                      val_output=(NUM_OUTPUT, ) + y_val_0.shape,
                      batch_size=BATCH_SIZE,
                      patience=PATIENCE,
                      lr=LEARNING_RATE)

tb = TensorBoard(log_dir=os.path.join(BASE_PATH, 'logs'))

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=2,
                   patience=PATIENCE)

cp = ModelCheckpoint(filepath=os.path.join(BASE_PATH, f'{MODEL_TYPE}_model_real.h5'),
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

model.save(os.path.join(BASE_PATH, f'{MODEL_TYPE}_model_real_final.h5'))

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
