import numpy as np

from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, BatchNormalization, Conv1D, MaxPooling1D, Input, Embedding, \
     Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam

from keras import backend as K
from keras import initializers

MAX_LEN_en = 3000
MAX_LEN_pr = 2000
NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('models/embedding_matrix.npy')


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
        # self.trainable_weights = [self.W, self.b, self.u]
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


def build_model() -> Model:
    enhancers = Input(shape=(MAX_LEN_en,))
    promoters = Input(shape=(MAX_LEN_pr,))

    emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(enhancers)
    emb_pr = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(promoters)

    enhancer_conv_layer = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(emb_en)
    enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(emb_pr)
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer)

    merge_layer = concatenate([enhancer_max_pool_layer, promoter_max_pool_layer], axis=1)
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    l_gru = Bidirectional(GRU(50, return_sequences=True))(dt)
    l_att = AttLayer(50)(l_gru)

    preds = Dense(1, activation='sigmoid')(l_att)

    model = Model([enhancers, promoters], preds)
    return model


if __name__ == '__main__':
    # 测试建立的模型是否正确
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    model.summary()
