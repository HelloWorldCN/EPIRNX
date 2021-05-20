import util

# 导入Keras
from keras import layers
from keras.layers import *
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

# 模型参数
enhancer_length = 3000
promoter_length = 2000

dense_layer_size = 256

weight_decay = 1e-5


class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        """

        :param attention_dim: attention_dim=50
        """
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        定义权重的地方
        :param input_shape: 输入数据维度，此处为 [batch_size, 246, 100]
        :return: 通过调用 super([Layer], self).build()完成。
        """
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='att_w')
        self.b = K.variable(self.init((self.attention_dim,)), name='att_b')
        self.u = K.variable(self.init((self.attention_dim, 1)), name='att_u')
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)  # 一定要在最后调用它

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        """
        编写层的功能逻辑的地方
        :param x:
        :param mask:
        :return:
        """
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        """
        如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。
        :param input_shape: 输入张量的形状
        :return: 改变后的张量形状
        """
        return (input_shape[0], input_shape[-1])


def grouped_convolution_block(init, grouped_channels, cardinality):
    # grouped_channels 每组的通道数
    # cardinality 多少组
    channel_axis = -1
    group_list = []
    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(init)
        x = Conv1D(filters=grouped_channels//4, kernel_size=10, padding='same')(x)
        x = Conv1D(filters=grouped_channels//2, kernel_size=5, padding='same')(x)
        x = Conv1D(filters=grouped_channels, kernel_size=2, padding='same')(x)
        x = BatchNormalization()(x)
        group_list.append(x)
    group_merge = concatenate(group_list, axis=channel_axis)
    return group_merge


def block_module(x, filters, kernel_size, cardinality):
    # residual connection
    X_shortcut = x
    X_shortcut = Conv1D(filters=filters*2, kernel_size=kernel_size, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # grouped_channels=256/8=下一层的filters=32,
    grouped_channels = int(filters / cardinality)

    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = grouped_convolution_block(x, grouped_channels=grouped_channels, cardinality=cardinality)
    x = Conv1D(filters=filters*2, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([X_shortcut, x])
    x = AveragePooling1D()(x)
    x = Activation('relu')(x)

    return x


def build_model(use_JASPAR=False):
    # 定义增强子分支
    enhancer_input = Input(shape=(enhancer_length, 4), name='enhancer_input')
    X = Conv1D(filters=32, kernel_size=40, padding='same')(enhancer_input)
    X = block_module(X, filters=256, kernel_size=20, cardinality=8)

    # 定义启动子分支
    promoter_input = Input(shape=(promoter_length, 4), name='promoter_input')
    Y = Conv1D(filters=32, kernel_size=40, padding='same')(promoter_input)
    Y = block_module(Y, filters=256, kernel_size=20, cardinality=8)

    # 将 增强子分支 和 启动子分支 组合在一起
    merge_layer = concatenate([X, Y], axis=1)

    # 定义混合在一起的层
    # Z = Flatten(name='all_flatten_1')(merge_layer)
    Z = Dense(units=128, activation='relu', name='all_dense_1')(merge_layer)
    Z = BatchNormalization(name='all_BN_1')(Z)
    Z = Dropout(0.5)(Z)

    Z = AttLayer(attention_dim=50, name='att')(Z)

    Z = Dense(units=1, activation='sigmoid')(Z)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=Z)

    return model


if __name__ == '__main__':
    # 测试建立的模型是否正确
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    model.summary()
