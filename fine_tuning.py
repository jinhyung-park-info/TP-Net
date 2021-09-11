import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from common.Utils import create_directory
from loss import *
from misc.model import *
import matplotlib.pyplot as plt
import numpy as np
from common.Utils import write_real_model_info
import random
from tensorflow.keras.models import load_model

# ========================= Model Constants ====================================

parser = argparse.ArgumentParser('Fine Tuning Model')
parser.add_argument('--epochs', required=False, default=200)
parser.add_argument('--batch_size', required=False, default=32)
parser.add_argument('--patience', required=False, default=15)
parser.add_argument('--lr', required=False, default=0.0001, help='Must be 1/10 of original lr')

# Loading Model Related
parser.add_argument('--model_type', required=False, default='global_pointnet', choices=['lstm', 'global_pointnet', 'local_pointnet'])
parser.add_argument('--seed', required=False, default=2)
parser.add_argument('--data_type', required=False, default='unordered', choices=['unordered', 'ordered', 'sorted'])
parser.add_argument('--data_offset', required=False, default=2)
parser.add_argument('--num_input_frames', required=False, default=5)
parser.add_argument('--num_output_frames', required=False, default=8)
parser.add_argument('--loss_type', required=False, default='chamfer', choices=['chamfer', 'mse', 'chamfer_and_shape', 'chamfer_and_mae'])
parser.add_argument('--retrain_scope', required=False, default='lstm', choices=['full', 'lstm', 'lstm_last', 'lstm + global extractor'])

# ========================= For Reproducibility ================================
FLAGS = parser.parse_args()
os.environ['PYTHONHASHSEED'] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)

BATCH_SIZE = int(FLAGS.batch_size)
EPOCHS = int(FLAGS.epochs)
PATIENCE = int(FLAGS.patience)
LEARNING_RATE = float(FLAGS.lr)

MODEL_TYPE = FLAGS.model_type
SEED = FLAGS.seed
DATA_TYPE = FLAGS.data_type
DATA_OFFSET = int(FLAGS.data_offset)
NUM_OUTPUT = int(FLAGS.num_output_frames)
NUM_INPUT = int(FLAGS.num_input_frames)
LOSS_TYPE = FLAGS.loss_type
RETRAIN_SCOPE = FLAGS.retrain_scope

BASE_PATH = create_directory(os.path.join('./result', MODEL_TYPE, f'{MODEL_TYPE}-{NUM_INPUT}', f'seed_{SEED}', 'fine_tuning'))

print('========================= Preparing Model =========================\n')

model = load_model(filepath=os.path.join('./result', MODEL_TYPE, f'{MODEL_TYPE}-{NUM_INPUT}', f'seed_{SEED}', f'{MODEL_TYPE}_model.h5'), compile=False)

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
    model.summary()
    single_frame_prediction_model.summary()

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

elif RETRAIN_SCOPE == 'lstm + global extractor':
    for layer in model.layers:
        if layer != model.get_layer('functional_3'):
            layer.trainable = False

    single_frame_prediction_model = model.get_layer('functional_3')

    for i, layer in enumerate(single_frame_prediction_model.layers):
        if i >= 10 or layer == single_frame_prediction_model.get_layer('functional_1'):
            layer.trainable = True
        else:
            layer.trainable = False

    global_extractor = single_frame_prediction_model.get_layer('functional_1')
    for layer in global_extractor.layers:
        if layer == global_extractor.get_layer('dense_4'):
            layer.trainable = True
        else:
            layer.trainable = False

    single_frame_prediction_model.summary()
    global_extractor.summary()
    model.summary()

else:
    for layer in model.layers:
        layer.trainable = True
    if MODEL_TYPE == 'global_pointnet':
        single_frame_prediction_model = model.get_layer('functional_3')
    else:
        single_frame_prediction_model = model.get_layer('functional_1')
        single_frame_prediction_model.summary()
        exit(0)


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

print('========================= Loading Data =========================\n')

x_train = np.load(f'data/real_world/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/x_train_pred_{DATA_TYPE}.npy')
y_train = np.load(f'data/real_world/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/y_train_pred_{DATA_TYPE}.npy')

x_val = np.load(f'data/real_world/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/x_val_pred_{DATA_TYPE}.npy')
y_val = np.load(f'data/real_world/preprocessed_data/offset_{DATA_OFFSET}_input_{NUM_INPUT}_output_{NUM_OUTPUT}/y_val_pred_{DATA_TYPE}.npy')

y_train = [y_train[i] for i in range(NUM_OUTPUT)]
y_val = [y_val[i] for i in range(NUM_OUTPUT)]

print('========================= Training Model =========================\n')

write_real_model_info(path=os.path.join(BASE_PATH),
                      model=single_frame_prediction_model,
                      simulation_seed=SEED,
                      retrain_scope=RETRAIN_SCOPE,
                      model_type=MODEL_TYPE,
                      data_type=DATA_TYPE,
                      loss_type=MODEL_TYPE,
                      offset=DATA_OFFSET,
                      train_input=x_train.shape,
                      train_output=(NUM_OUTPUT, ) + y_train[0].shape,
                      val_input=x_val.shape,
                      val_output=(NUM_OUTPUT, ) + y_val[0].shape,
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
