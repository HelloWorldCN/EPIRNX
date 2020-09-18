import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv1D, Activation, MaxPooling1D, Input, \
    Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

sample_num = 44313

# 模型参数
enhancer_length = 3000  # TODO: get this from input
promoter_length = 2000  # TODO: get this from input
filters_slim = 256
filters = 512  # Number of kernels; used to be 1024
kernel_slim_size = 20
kernel_size = 40  # Length of each kernel
LSTM_out_dim = 20  # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 256


def identity_block(X, base_name):
    X_shortcut = X  # 保存输入

    # 第一层
    X = Conv1D(filters=filters_slim, kernel_size=kernel_size, padding='valid', kernel_regularizer=l2(1e-5), name=base_name+'_Conv_1')(X)
    X = BatchNormalization(name=base_name+'_BN_1')(X)
    X = Activation('relu')(X)

    # 第二层
    X = Conv1D(filters=filters, kernel_size=kernel_slim_size, padding="same", kernel_regularizer=l2(1e-5), name=base_name+'_Conv_2')(X)
    X = BatchNormalization(name=base_name+'_BN_2')(X)

    # 处理输入数据
    X_shortcut = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', kernel_regularizer=l2(1e-5), name=base_name+'_Conv_3')(X_shortcut)
    X_shortcut = BatchNormalization(name=base_name+'_BN_3')(X_shortcut)

    # 跳跃连接
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    # 进一步减小维度
    X = Conv1D(filters=1024, kernel_size=10, strides=10, padding='valid', kernel_regularizer=l2(1e-5), name=base_name+'_Conv_4')(X)

    # 池化
    X = MaxPooling1D(name=base_name + '_MaxPooling_1')(X)

    return X


def build_model() -> Model:
    # 定义增强子分支
    enhancer_input = Input(shape=(enhancer_length, 4), name='enhancer_input')
    X = Conv1D(filters=1024, kernel_size=40, padding='same')(enhancer_input)
    X = MaxPooling1D(strides=20)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)


    # 定义启动子分支
    promoter_input = Input(shape=(promoter_length, 4), name='promoter_input')
    Y = Conv1D(filters=1024, kernel_size=40, padding='same')(promoter_input)
    Y = MaxPooling1D(strides=20)(Y)
    Y = BatchNormalization()(Y)
    Y = Dropout(0.5)(Y)

    # 将 增强子分支 和 启动子分支 组合在一起
    merge_layer = concatenate([X, Y], axis=1)

    Z = LSTM(units=LSTM_out_dim)(merge_layer)
    Z = BatchNormalization()(Z)
    Z = Dropout(0.5)(Z)

    Z = Dense(925, activation='relu')(Z)
    Z = BatchNormalization()(Z)
    Z = Dropout(0.5)(Z)

    Z = Dense(1, activation='sigmoid', name='dense_99')(Z)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=Z)

    return model


if __name__ == '__main__':
    # 测试建立的模型是否正确
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    model.summary()
