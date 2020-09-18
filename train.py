import os
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from utility import load_train_data, load_test_data, plot_history, recall, precision, f1


model_names = ['EPIVAN', 'CBCNN', 'ResNeXt', 'SPEID']
dataset_types = ['balance', 'imbalance']
dataset_codings = ['onehot', 'embedding']
cell_lines = ['GM12878', 'HeLa-S3', 'K562', 'IMR90', 'NHEK', 'HUVEC']


# Script arguments.
parser = argparse.ArgumentParser(description='EPIRNX')
parser.add_argument('--model-name', metavar='str', help='The name of model.', default='ResNeXt')
parser.add_argument('--dataset-dir', metavar='dir', help='Dataset root path. For example:F:/Data/EPIs/', default='F:/Data/EPIs/')
parser.add_argument('--dataset-type', metavar='str', help='balance or imbalance.', default='imbalance')
parser.add_argument('--dataset-coding', metavar='str', help='onehot or embedding.', default='onehot')
parser.add_argument('--cell-line', metavar='str', help='The cell-line which you want to Train.', default='GM12878')
parser.add_argument('--batch-size', metavar='int', default=32)
parser.add_argument('--epochs', metavar='int', help='Num of training iterations.', default=25)
parser.add_argument('--train-verbose', metavar='int', help='0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.', default=1)


def _get_model(model_name: str) -> Model:
    bm = None
    if model_name == 'SPEID':
        import model_onehot_SPEID as bm
    elif model_name == 'CBCNN':
        import model_onehot_CBCNN as bm
    elif model_name == 'EPIVAN':
        import model_embedding_EPIVAN as bm
    elif model_name == 'ResNeXt':
        import model_onehot_ResNeXts as bm
    model = bm.build_model()

    return model


def main(model_name: str,
         dataset_dir: str,
         dataset_type: str,
         dataset_coding: str,
         cell_line: str,
         batch_size: int,
         epochs: int,
         train_verbose: int,
         save_model: bool = True):

    # get data
    X_en_tra, X_pr_tra, y_tra = load_train_data(dataset_dir, dataset_type, dataset_coding, cell_line)
    X_en_val, X_pr_val, y_val = load_test_data(dataset_dir, dataset_coding, cell_line)

    # get model
    model = _get_model(model_name)
    opt = Adam(learning_rate=1e-4, decay=0.001)  # opt = RMSprop(lr = 1e-6), Adam(lr=1e-5) TODO 从参数接收
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', recall, precision, f1])
    model.summary()

    # 开始训练
    print('start train, dataset is {}_{}_{}, model is {}, epochs is {}...'.format(cell_line, dataset_coding, dataset_type, model_name, epochs))
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    history = model.fit(x=[X_en_tra, X_pr_tra], y=y_tra,
                        validation_data=([X_en_val, X_pr_val], y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=train_verbose
                        )

    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    use_time = (end_time-start_time).total_seconds() / 60
    print('training accomplishment! start_time: {}, end_time: {}, use {} minutes'.format(start_time_str, end_time_str, use_time))
    print('starting plot history...')
    plot_history(cell_line, history, dataset_type, dataset_coding, model_name)
    print('plot history accomplishment!')

    if save_model:
        print("save model to models/weights/train_on_{}_{}_{}_{}_weights.hfd5".format(dataset_type, cell_line, dataset_coding, model_name))
        model.save_weights('models/weights/train_on_{}_{}_{}_{}_weights.hfd5'.format(dataset_type, cell_line, dataset_coding, model_name))  # 使用model.load_weights('')读取权重


if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.model_name
    dataset_dir = args.dataset_dir
    dataset_type = args.dataset_type
    dataset_coding = args.dataset_coding
    cell_line = args.cell_line
    batch_size = args.batch_size
    epochs = args.epochs
    train_verbose = args.train_verbose

    print('model_name:{}, dataset_dir:{}, dataset_type:{}, dataset_coding:{}, cell_line:{}, batch_size:{}, epochs:{}, train_verbose:{}'.format(model_name, dataset_dir, dataset_type, dataset_coding, cell_line, batch_size, epochs, train_verbose))

    assert model_name in model_names
    assert dataset_type in dataset_types
    assert dataset_coding in dataset_codings
    assert cell_line in cell_lines


    main(model_name, dataset_dir, dataset_type, dataset_coding, cell_line, batch_size, epochs, train_verbose, save_model=True)



