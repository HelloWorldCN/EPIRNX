import tensorflow as tf
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, BatchNormalization, Conv1D, Activation, add, MaxPooling1D, Input, \
    Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 模型参数
enhancer_length = 3000
promoter_length = 2000

filters_1 = 128
filters_2 = 256

kernel_size_1 = 40
kernel_size_2 = 20

dense_layer_size = 256

weight_decay = 1e-5


def grouped_convolution_block(init, grouped_channels, kernel_size, cardinality, strides, base_name):
    stage = 2
    # grouped_channels 每组的通道数
    # cardinality 多少组
    channel_axis = -1
    group_list = []
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels], name=base_name+'_stage_'+str(stage)+'_Lambda_'+str(c))(init)
        x = Conv1D(filters=grouped_channels, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name=base_name+'_stage_'+str(stage)+'_Conv_'+str(c))(x)
        group_list.append(x)
    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)
    return x


def block_module(x, filters, kernel_size, cardinality, base_name):
    stage = 1

    # residual connection
    X_shortcut = x
    X_shortcut = Conv1D(filters=(filters * 2), kernel_size=(kernel_size + 10), strides=10, padding='valid', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name=base_name + '_stage_' + str(stage) + '_Conv_0')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # grouped_channels=下一层的filters=32,
    grouped_channels = int(filters / cardinality)

    # 如果没有down sampling就不需要这种操作
    # if init._keras_shape[-1] != 2 * filters:
    #     init = Conv1D(filters=(filters * 2), kernel_size=1, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
    #     init = BatchNormalization()(init)

    # conv1
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name=base_name+'_stage_'+str(stage)+'_Conv_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv2(group)，选择在 group 的时候 down sampling
    # grouped_channels=下一层的filters=32, cardinality=多少组=4
    x = grouped_convolution_block(x, grouped_channels=grouped_channels, kernel_size=20, cardinality=cardinality, strides=1, base_name=base_name)

    # conv3
    x = Conv1D(filters=(filters * 2), kernel_size=10, strides=10, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name=base_name+'_stage_'+str(stage)+'_Conv_2')(x)
    x = BatchNormalization()(x)

    x = add([X_shortcut, x])
    x = MaxPooling1D()(x)
    x = Activation('relu')(x)

    return x


def build_model() -> Model:
    # 定义增强子分支
    enhancer_input = Input(shape=(enhancer_length, 4), name='enhancer_input')
    X = block_module(enhancer_input, filters=filters_1, kernel_size=40, cardinality=4, base_name='enhancer')

    # 定义启动子分支
    promoter_input = Input(shape=(promoter_length, 4), name='promoter_input')
    Y = block_module(promoter_input, filters=filters_1, kernel_size=40, cardinality=4, base_name='promoter')

    # 将 增强子分支 和 启动子分支 组合在一起
    merge_layer = concatenate([X, Y], axis=1)

    # 定义混合在一起的层
    Z = Flatten(name='all_flatten_1')(merge_layer)
    Z = Dense(units=dense_layer_size, activation='relu', kernel_regularizer=l2(1e-6), name='all_dense_1')(Z)
    Z = BatchNormalization(name='all_BN_1')(Z)
    Z = Dropout(0.5)(Z)
    Z = Dense(units=1, activation='sigmoid')(Z)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=Z)

    return model


if __name__ == '__main__':
    # 测试建立的模型是否正确
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    model.summary()
