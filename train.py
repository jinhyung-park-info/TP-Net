from config import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from data.data_loader import *
from prediction_models import *
import matplotlib.pyplot as plt


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_filepath):
        tf.keras.callbacks.Callback.__init__(self)
        self.log_filepath = os.path.join(log_filepath, 'train_log.txt')

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        with open(self.log_filepath, 'a') as file:
            file.write(f'Epoch {epoch}:  ')
            file.write('%.5f' % loss)
            file.write('  ')
            file.write('%.5f' % val_loss)
            file.write('\n')


if __name__ == '__main__':
    args = gen_args()
    set_seed(args.seed)

    print('========================= Loading Data =========================\n')
    x_train, y_train, x_val, y_val = get_data(args)

    print('========================= Building Model =========================\n')
    model = model_builder(args)

    print(str(args))
    with open(os.path.join(args.save_path, 'train_log.txt'), 'w') as f:
        f.write('------------- Parameters ------------\n')
        f.write(str(args))
        f.write('\n\n\n')

        f.write('------------- Loss ------------\n')

    tb = TensorBoard(log_dir=os.path.join(args.save_path, 'logs'))

    cp = ModelCheckpoint(filepath=os.path.join(args.save_path, f'{args.model_type}_best.h5'),
                         save_weights_only=False,
                         monitor='val_loss',
                         mode='min',
                         save_best_only=True,
                         save_freq='epoch',
                         verbose=args.cp_verbose)

    log_callback = LogCallback(args.save_path)

    print('========================= Training Model =========================\n')
    history = model.fit(x_train, y_train,
                        epochs=args.epoch,
                        batch_size=args.batch_size,
                        verbose=args.train_verbose,
                        validation_data=(x_val, y_val),
                        callbacks=[tb, cp, log_callback])

    model.save(os.path.join(args.save_path, f'{args.model_type}_epoch_{args.epoch}.h5'))

    print('========================= Evaluating Model =========================\n')
    train_evaluation = model.evaluate(x_train, y_train, batch_size=args.batch_size)
    val_evaluation = model.evaluate(x_val, y_val, batch_size=args.batch_size)

    print(f'Train Loss: {str(train_evaluation)}')
    print(f'Validation Loss: {str(val_evaluation)}')

    with open(os.path.join(args.save_path, 'train_log.txt'), 'a') as f:
        f.write('\n\n------------- Evaluation ------------\n')
        f.write(f'Train Loss: {str(train_evaluation)}\n')
        f.write(f'Validation Loss: {str(val_evaluation)}\n')

    print('========================= Plotting Loss Graph =========================\n')
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))

    # Loss Graph
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, 'loss.png'), dpi=600)

    # Loss Graph after 5 epochs
    if args.epoch > 5:
        plt.clf()
        plt.plot(epochs[5:], train_loss[5:], label='train')
        plt.plot(epochs[5:], val_loss[5:], label='val')
        plt.legend()
        plt.savefig(os.path.join(args.save_path, 'loss_zoom.png'), dpi=600)
