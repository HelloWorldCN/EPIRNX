import os
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from utility import DataGenerator, get_partition, plot_history, recall, precision, f1


model_names = ['ResNeXt']
dataset_types = ['balance', 'imbalance']
dataset_codings = ['onehot', 'embedding']
cell_lines = ['GM12878', 'HeLa-S3', 'K562', 'IMR90', 'NHEK', 'HUVEC']

# Script arguments.
parser = argparse.ArgumentParser(description='Transfer EPIRNX')
parser.add_argument('--model-name', metavar='str', help='The name of model', default='ResNeXt')
parser.add_argument('--dataset-dir', metavar='dir', help='Dataset root path. For example:G:/Data/EPIs/Data_bal_By_IDs/', default='G:/Data/EPIs/Data_bal_By_IDs/')
parser.add_argument('--dataset-type', metavar='str', help='balance or imbalance.', default='imbalance')
parser.add_argument('--dataset-coding', metavar='str', help='onehot or embedding.', default='onehot')
parser.add_argument('--cell-line', metavar='str', help='The cell-line of the pre-train model.', default='GM12878')
parser.add_argument('--batch-size', metavar='int', default=64)
parser.add_argument('--epochs', metavar='int', help='Num of training iterations', default=20)
parser.add_argument('--train-verbose', metavar='int', help='0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.', default=1)


def _get_model(model_name: str = 'ResNeXt') -> Model:
    bm = None
    if model_name == 'ResNeXt':
        import model_onehot_ResNeXts as bm
    model = bm.build_model()

    return model


def main(model_name: str,
         dataset_dir: str,
         dataset_type: str,
         dataset_coding: str,
         cell_line: str,  # pre-train cell-line
         epochs: int,
         train_verbose: int,
         save_model: bool = True):
    # Parameters passed to the Generator
    params = {'batch_size': batch_size,
              'enhancer_dim': (3000, 4),
              "promoter_dim": (2000, 4),
              "shuffle": True}

    # partition dict, which contains the IDs of train/validation dataset storage on local disk
    partition_file_path = dataset_dir + 'partition.json'
    partition = get_partition(pre_train_cell_line=cell_line, partition_file_path=partition_file_path)

    # DataGenerator
    training_generator = DataGenerator(data_dir=dataset_dir, list_IDs=partition['train'], **params)
    validation_generator = DataGenerator(data_dir=dataset_dir, list_IDs=partition['validation'], **params)

    # get pre-train model
    model = _get_model(model_name)
    # load weights
    weight_path = 'models/weights/train_on_{}_{}_{}_{}_weights.hfd5'.format(dataset_type, cell_line, dataset_coding, model_name)
    model.load_weights(weight_path)

    # set Trainable layers
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    model.layers[-2].trainable = True
    model.layers[-3].trainable = True
    model.layers[-4].trainable = True

    opt = Adam(learning_rate=1e-4, decay=0.001)  # opt = RMSprop(lr = 1e-6), Adam(lr=1e-5) TODO 从参数接收
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', recall, precision, f1])
    model.summary()

    # trainable layers
    for x in model.trainable_weights:
        print("可训练层：{0}".format(x.name))
    print('\n')

    # non-trainable layers
    for x in model.non_trainable_weights:
        print("不可训练：{0}".format(x.name))
    print('\n')

    # start train
    print('start train, dataset is {}_{}_{}, pre-train model is {}, epochs is {}...'.format(cell_line, dataset_coding, dataset_type, model_name, epochs))
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  verbose=train_verbose)

    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    use_time = (end_time - start_time).total_seconds() / 60
    print('training accomplishment! start_time: {}, end_time: {}, use {} minutes'.format(start_time_str, end_time_str,
                                                                                         use_time))
    print('starting plot history...')
    plot_history(cell_line, history, dataset_type, dataset_coding, model_name)
    print('plot history accomplishment!')

    if save_model:
        print("save model to models/weights/train_on_{}_{}_{}_{}_Transfer_weights.hfd5".format(dataset_type, cell_line, dataset_coding, model_name))
        model.save_weights('models/weights/train_on_{}_{}_{}_{}_Transfer_weights.hfd5'.format(dataset_type, cell_line, dataset_coding, model_name))  # 使用model.load_weights('')读取权重


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

    print('model_name:{}, dataset_dir:{}, dataset_type:{}, dataset_coding:{}, pre_train_cell_line:{}, batch_size:{}, epochs:{}, train_verbose:{}'.format(model_name, dataset_dir, dataset_type, dataset_coding, cell_line, batch_size, epochs, train_verbose))

    assert model_name in model_names
    assert dataset_type in dataset_types
    assert dataset_coding in dataset_codings
    assert cell_line in cell_lines

    main(model_name, dataset_dir, dataset_type, dataset_coding, cell_line, epochs, train_verbose, save_model=True)

