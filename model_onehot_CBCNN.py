import tensorflow as tf
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPooling1D, Input, \
    Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 模型参数
enhancer_length = 3000  # TODO: get this from input
promoter_length = 2000  # TODO: get this from input
filters_1 = 300
kernel_size_1 = 40
dense_layer_size = 800
weight_decay = 1e-5

def build_model() -> Model:
    # 定义增强子分支
    enhancer_input = Input(shape=(enhancer_length, 4), name='enhancer_input')
    X = Conv1D(filters=filters_1, kernel_size=kernel_size_1, padding='same')(enhancer_input)
    X = MaxPooling1D(pool_size=20, strides=20)(X)

    # 定义启动子分支
    promoter_input = Input(shape=(promoter_length, 4), name='promoter_input')
    Y = Conv1D(filters=filters_1, kernel_size=kernel_size_1, padding='same')(promoter_input)
    Y = MaxPooling1D(pool_size=20, strides=20)(Y)

    # 将 增强子分支 和 启动子分支 组合在一起
    merge_layer = concatenate([X, Y], axis=1)

    # 定义混合在一起的层
    Z = Flatten(name='all_flatten_1')(merge_layer)
    Z = Dense(units=dense_layer_size, activation='relu', kernel_regularizer=l2(1e-6), name='all_dense_1')(Z)
    Z = BatchNormalization(name='all_BN_1')(Z)
    Z = Dropout(0.2)(Z)
    Z = Dense(units=1, activation='sigmoid')(Z)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=Z)

    return model


if __name__ == '__main__':
    # 测试建立的模型是否正确
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    model.summary()
