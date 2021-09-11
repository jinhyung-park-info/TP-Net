from config import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from data.data_loader import *
from prediction_models import *


if __name__ == '__main__':
    args = gen_args()
    set_seed(args.seed)     # set random seeds for reproducibility

    print('========================= Loading Data =========================\n')
    x_train, y_train, x_val, y_val = get_data(args)

    print('========================= Building Model =========================\n')
    model = model_builder(args)
    print(str(args))

    tb = TensorBoard(log_dir=os.path.join(args.save_path, 'logs'))

    cp = ModelCheckpoint(filepath=os.path.join(args.save_path, f'tp_net_best.h5'),
                         save_weights_only=False,
                         monitor='val_loss',
                         mode='min',
                         save_best_only=True,
                         save_freq='epoch',
                         verbose=args.cp_verbose)

    print('========================= Training Model =========================\n')
    history = model.fit(x_train, y_train,
                        epochs=args.epoch,
                        batch_size=args.batch_size,
                        verbose=args.train_verbose,
                        validation_data=(x_val, y_val),
                        callbacks=[tb, cp])

    model.save(os.path.join(args.save_path, f'tp_net_epoch_{args.epoch}.h5'))

    print('========================= Evaluating Model =========================\n')
    train_evaluation = model.evaluate(x_train, y_train, batch_size=args.batch_size)
    val_evaluation = model.evaluate(x_val, y_val, batch_size=args.batch_size)

    print(f'Train Loss: {str(train_evaluation)}')
    print(f'Validation Loss: {str(val_evaluation)}')

    with open(os.path.join(args.save_path, 'train_log.txt'), 'w') as f:
        f.write('------------- Evaluation ------------\n')
        f.write(f'Train Loss: {str(train_evaluation)}\n')
        f.write(f'Validation Loss: {str(val_evaluation)}\n')

    print('========================= Done =========================\n')
