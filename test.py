import os
import argparse

import tensorflow as tf
from tensorflow.keras import Model

from utility import load_test_data, evaluate_metrics


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
parser.add_argument('--train-cell-line', metavar='str', help='The train-cell-line of the model.', default='GM12878')
parser.add_argument('--batch-size', metavar='int', help='batch-size', default=32)
parser.add_argument('--test-verbose', metavar='int', help='0 or 1 or 2, 0=silent, 1=progress bar, 2=one line per epoch.', default=1)


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
         train_cell_line: str,
         batch_size: int,
         test_verbose: int):

    # get model
    model = _get_model(model_name)
    model.summary()
    # load weights
    model.load_weights('models/weights/train_on_{}_{}_{}_{}_weights.hfd5'.format(dataset_type, train_cell_line, dataset_coding, model_name))

    print('start test, model\'s training dataset is {}_{}_{}, model is {}...'.format(dataset_type, train_cell_line, dataset_coding, model_name))
    for test_cell_line in cell_lines:  # test all cell-lines
        # get test_cell_line data
        X_en_tes, X_pr_tes, y_tes = load_test_data(dataset_dir, dataset_coding, test_cell_line)
        # get predict results
        y_hat = model.predict([X_en_tes, X_pr_tes], batch_size=batch_size, verbose=test_verbose)

        print('test_cell_line is {}'.format(test_cell_line))
        results = evaluate_metrics(y_tes, y_hat)
        print('auROC={}, auPR={}, precision={}, \
              recall={}, f1_score={}, acc={}\n'
              .format(results['auROC'], results['auPR'], results['precision'],
                      results['recall'], results['f1_score'], results['acc']))



if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.model_name
    dataset_dir = args.dataset_dir
    dataset_type = args.dataset_type
    dataset_coding = args.dataset_coding
    train_cell_line = args.train_cell_line
    batch_size = args.batch_size
    test_verbose = args.test_verbose

    print('model_name:{}, dataset_dir:{}, dataset_type:{}, dataset_coding:{}, cell_line:{}, batch_size:{}, epochs:{}, train_verbose:{}'.format(model_name, dataset_dir, dataset_type, dataset_coding, cell_line, batch_size, epochs, train_verbose))

    assert model_name in model_names
    assert dataset_type in dataset_types
    assert dataset_coding in dataset_codings
    assert train_cell_line in cell_lines

    main(model_name, dataset_dir, dataset_type, dataset_coding, train_cell_line, batch_size, test_verbose)
