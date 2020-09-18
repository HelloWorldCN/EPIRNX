import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn import metrics

import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import History


def load_train_data(dataset_dir: str,
                    dataset_type: str,
                    dataset_coding: str,
                    cell_line: str):
    print('从 {} 加载 {}_{} 的 {} 细胞系训练数据'.format(dataset_dir, dataset_coding, dataset_type, cell_line))

    if dataset_coding == 'onehot':
        if dataset_type == 'balance':
            train_dataset_dir = dataset_dir + 'onehot/{}_onehot_train.npz'.format(cell_line)  # onehot的平衡数据集
        else:
            train_dataset_dir = dataset_dir + 'onehot/{}_onehot_imtrain.npz'.format(cell_line)  # onehot的不平衡数据集
    else:
        if dataset_type == 'balance':
            train_dataset_dir = dataset_dir + 'embedding/{}_train.npz'.format(cell_line)  # embedding的平衡数据集
        else:
            train_dataset_dir = dataset_dir + 'embedding/{}_imtrain.npz'.format(cell_line)  # embedding的平衡数据集

    train_data = np.load(train_dataset_dir)
    X_en_tra, X_pr_tra, y_tra = train_data['X_en_tra'], train_data['X_pr_tra'], train_data['y_tra']
    # print("训练数据统计：")
    # print("X_en_tra.shape:{}, X_pr_tra.shape:{}, y_tra.shape:{}".format(X_en_tra.shape, X_pr_tra.shape, y_tra.shape))
    return X_en_tra, X_pr_tra, y_tra


def load_test_data(dataset_dir: str,
                   dataset_coding: str,
                   cell_line: str):
    print('从 {} 加载 {} 的 {} 细胞系测试数据'.format(dataset_dir, dataset_coding, cell_line))

    if dataset_coding == 'onehot':
        val_dataset_dir = dataset_dir + 'onehot/{}_onehot_test.npz'.format(cell_line)  # onehot的测试集
    else:
        val_dataset_dir = dataset_dir + 'embedding/{}_test.npz'.format(cell_line)  # embedding的测试集

    val_data = np.load(val_dataset_dir)
    X_en_val, X_pr_val, y_val = val_data['X_en_val'], val_data['X_pr_val'], val_data['y_val']
    # print("测试数据统计：")
    # print("X_en_val.shape:{}, X_pr_val.shape:{}, y_val.shape:{}".format(X_en_val.shape, X_pr_val.shape, y_val.shape))
    return X_en_val, X_pr_val, y_val


class DataGenerator(Sequence):
    def __init__(self, data_dir, list_IDs, batch_size=32, enhancer_dim=(3000, 4), promoter_dim=(2000, 4), shuffle=True):
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.enhancer_dim = enhancer_dim
        self.promoter_dim = promoter_dim
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        产生一个batch的数据
        :param index:
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X_en_tra, X_pr_tra, y_tra = self.__data_generation(list_IDs_temp)

        return [X_en_tra, X_pr_tra], y_tra

    def __data_generation(self, list_IDs_temp):
        """
        'Generates data containing batch_size samples' # X_en_tra : (n_samples, *dim, n_channels)
        :param list_IDs_temp:
        :return:
        """
        # Initialization
        X_en_tra, X_pr_tra, y_tra = np.empty(shape=(self.batch_size, *self.enhancer_dim)), np.empty(shape=(self.batch_size, *self.promoter_dim)), np.empty(shape=(self.batch_size,))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            datum = np.load(self.data_dir + str(ID) + '.npz')
            X_en_tra_temp = datum["enhancer"]
            X_pr_tra_temp = datum["promoter"]
            y_tra_temp = datum["label"]

            X_en_tra[i, ] = X_en_tra_temp
            X_pr_tra[i, ] = X_pr_tra_temp
            y_tra[i] = y_tra_temp

        return X_en_tra, X_pr_tra, y_tra


def get_partition(pre_train_cell_line: str,
                  partition_file_path: str = 'G:/Data/EPIs/Data_bal_By_IDs/partition.json') -> dict:
    # 从本地加载所有数据的ID
    all_data_indexes = __read_json_data(partition_file_path)

    all_data_train = all_data_indexes['train']
    all_data_validation = all_data_indexes['validation']

    partition = {'train': [],
                 'validation': []}

    for partition_train_cell_line in all_data_train:
        # 迁移学习，去掉原本的训练集
        if partition_train_cell_line == pre_train_cell_line:
            continue
        else:
            partition_train_tmp = all_data_train[partition_train_cell_line]
            partition['train'].extend(partition_train_tmp)

    for partition_val_cell_line in all_data_validation:
        partition_val_tmp = all_data_validation[partition_val_cell_line]
        partition['validation'].extend(partition_val_tmp)

    transform_train_data_num = len(partition['train'])
    transform_val_data_num = len(partition['validation'])
    print("迁移学习中，除去 {} 之后的训练数据为 {} 条，验证数据为 {} 条".format(pre_train_cell_line, transform_train_data_num, transform_val_data_num))

    return partition


def __read_json_data(file_path):
    """
    读取JSON数据
    :param file_path: json文件名称
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as rf:
        # 将JSON对象转换为字典
        json_dict = json.loads(rf.read())
        return json_dict


def plot_history(cell_line: str,
                 history: History,
                 dataset_type: str,
                 dataset_coding: str,
                 model_type: str):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f1 = history.history['f1']
    val_f1 = history.history['val_f1']
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']

    plt.figure()
    plt.title('Train on {} Cell-line\nAccuracy'.format(cell_line))
    plt.plot(accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Val Accuracy')
    plt.legend()
    # 保存到本地，例如，Train_on_GM12878_bal_onehot_Normal_Accuracy.png
    plt.savefig('D:/Experiment_Image/Train_on_{}_{}_{}_{}_Accuracy.png'.format(cell_line, dataset_type, dataset_coding,
                                                                               model_type))

    plt.figure()
    plt.title('Train on {} Cell-line\nLoss'.format(cell_line))
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.savefig(
        'D:/Experiment_Image/Train_on_{}_{}_{}_{}_Loss.png'.format(cell_line, dataset_type, dataset_coding, model_type))

    plt.figure()
    plt.title('Train on {} Cell-line\nF1, Recall, Precision'.format(cell_line))
    plt.plot(f1, label='f1')
    plt.plot(val_f1, label='val_f1')
    plt.plot(precision, label='precision')
    plt.plot(val_precision, label='val_precision')
    plt.plot(recall, label='recall')
    plt.plot(val_recall, label='val_recall')
    plt.legend()
    plt.savefig(
        'D:/Experiment_Image/Train_on_{}_{}_{}_{}_FRP.png'.format(cell_line, dataset_type, dataset_coding, model_type))

    plt.show()
    # plt.close()


def evaluate_metrics(y_tes, y_hat):
    y_hat_01 = np.array([1. if y >= 0.5 else 0. for y in y_hat])
    # ------------------使用sklearn.metrics计算------------------
    auROC = metrics.roc_auc_score(y_tes, y_hat)
    p, r, _, = metrics.precision_recall_curve(y_tes, y_hat)
    auPR = -np.trapz(p, r)
    f1_score = metrics.f1_score(y_tes, y_hat_01)
    acc = metrics.accuracy_score(y_tes, y_hat_01)
    # ------------------End------------------
    return {'precision': p,
            'recall': r,
            'auROC': auROC,
            'auPR': auPR,
            'f1_score': f1_score,
            'acc': acc}


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))
