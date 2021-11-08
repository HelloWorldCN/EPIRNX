import tensorflow as tf
import os
import random
from dataLoader import DataLoader, Vocab


class Generator:

    def __init__(self, dataloader: DataLoader, vocab: Vocab, batch_size, embedding_dim, seq_length, checkpoint_dir, rnn_units=32, start_token=0, learning_rate=1e-4):
        self.dataloader = dataloader
        self.vocab = vocab
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.vocab_size = self.vocab.vocab_size
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "gen/my_ckpt")
        # self.checkpoint_prefix = checkpoint_dir
        # self.start_token = tf.identity(tf.constant([start_token] * self.batch_size, dtype=tf.int32))
        self.start_token = tf.identity(tf.constant(random.choices(self.vocab.vocab_ids, k=self.batch_size), dtype=tf.int32))

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.model = tf.keras.Sequential([
            # Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[batch_size, None]),

            # Layer 2: LSTM with `rnn_units` number of units.
            # tf.keras.layers.LSTM(
            #     units=rnn_units,
            #     return_sequences=False,
            #     recurrent_initializer='glorot_uniform',
            #     recurrent_activation='sigmoid',
            #     stateful=False,  # TODO this was true. why?
            #     dropout=0.4
            # ),

            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rnn_units, return_sequences=False)),

            tf.keras.layers.Dense(self.vocab_size)
        ])
        self.model.summary()


        self.g_embeddings = next(layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Embedding)).embeddings

    def load_weights(self):
        print(self.checkpoint_prefix)
        # print(os.listdir(self.checkpoint_prefix))
        try:
            self.model.load_weights(self.checkpoint_prefix)
            print('loaded weights for generator')
        except:
            print('could not find weights to load for generator')

    @tf.function
    def generate(self, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_length
        gen_x = tf.TensorArray(dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, gen_x):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.model(x_t)
            log_prob = tf.math.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = next_token
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, gen_x

        gen_x = gen_x.write(0, self.start_token)

        _, _, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2: i < seq_len,
            body=_g_recurrence,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       self.start_token,
                       gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length
        self.model.reset_states()
        return self.gen_x

    @tf.function
    def gen_predictions(self, x, training=False):  # x in token form [batch_size, seq_length]
        # supervised pretraining for generator
        g_predictions = tf.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
        x_transposed = tf.cast(tf.transpose(x), dtype=tf.int32)
        ta_x = tf.TensorArray(dtype=tf.int32, size=self.seq_length)
        ta_x = ta_x.unstack(x_transposed)

        def _pretrain_recurrence(i, x_t, g_predictions):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.model(x_t, training=training)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_x.read(i)
            return i + 1, x_tp1, g_predictions

        ta_x.write(0, self.start_token)
        _, _, self.g_predictions = tf.while_loop(
            cond=lambda i, _1, _2: i < self.seq_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       ta_x.read(0),
                       g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.model.reset_states()
        return self.g_predictions

    @tf.function
    def train_step(self, samples, rewards):

        with tf.GradientTape() as tape:
            loss = self.get_loss(samples, rewards)

        g_grad, _ = tf.clip_by_global_norm(tape.gradient(loss, self.model.trainable_variables), 5.0)
        g_updates = self.optimizer.apply_gradients(zip(g_grad, self.model.trainable_variables))

        return loss

    def test_step(self):
        x, y = self.dataloader.get_batch(shuffle=True, seq_length=self.seq_length, batch_size=self.batch_size, training=False)
        y_hat = self.gen_predictions(tf.constant(x))
        gen_loss = self.get_pretrain_loss(labels=x, samples=y_hat)
        return gen_loss

    @tf.function
    def get_pretrain_loss(self, labels, samples):  # labels as tokens, samples as prob distr
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, samples, from_logits=False)
        return loss

    @tf.function
    def get_loss(self, x, rewards):
        g_predictions = self.gen_predictions(x)
        loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.cast(tf.reshape(x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0)
                *
                tf.math.log(tf.clip_by_value(tf.reshape(g_predictions, [-1, self.vocab_size]), 1e-20, 1.0))
                # tf.clip_by_value()功能：可以将一个张量中的数值限制在一个范围之内。（可以避免一些运算错误:可以保证在进行log运算时，不会出现log0这样的错误或者大于1的概率）
                , 1)
            * tf.reshape(rewards, [-1])
        )
        return loss