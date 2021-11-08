import tensorflow as tf
import numpy as np
from dataLoader import Vocab

import random


class Rollout(object):
    def __init__(self, generator, discriminator, vocab: Vocab, batch_size, seq_length, rollout_num=5):
        self.generator = generator
        self.discriminator = discriminator
        self.vocab = vocab
        self.batch_size = batch_size
        self.seq_length = seq_length
        # self.start_token = tf.identity(tf.constant([start_token] * self.batch_size, dtype=tf.int32))
        self.start_token = tf.identity(tf.constant(random.choices(self.vocab.vocab_ids, k=self.batch_size), dtype=tf.int32))
        self.rollout_num = rollout_num

    @tf.function
    def autobots(self, given_num, x):
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.seq_length, dynamic_size=False, infer_shape=True)
        x = tf.cast(x, tf.int32)
        ta_x = tf.TensorArray(dtype=tf.int32, size=self.seq_length)
        ta_x = ta_x.unstack(tf.transpose(x, perm=[1, 0]))

        # 如果i < given_num
        def _g_recurrent_1(i, x_t, given_num, gen_x):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            _ = self.generator.model(x_t)
            x_tp_1 = ta_x.read(i)
            gen_x = gen_x.write(i, x_tp_1)
            return i + 1, x_tp_1, given_num, gen_x

        def _g_recurrent_2(i, x_t, given_num, gen_x):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.generator.model(x_t)
            log_prob = tf.math.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp_1 = next_token
            gen_x = gen_x.write(i, next_token)
            return i + 1, x_tp_1, given_num, gen_x

        gen_x.write(0, self.start_token)

        # 从前面状态x中读取token
        i, x_t, given_num, self.gen_x = tf.while_loop(
            cond=lambda i, _1, given_num, _2: i < given_num,
            body=_g_recurrent_1,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       self.start_token,
                       given_num,
                       gen_x)
        )

        # 生成后面的token
        _, _, _, self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.seq_length,
            body=_g_recurrent_2,
            loop_vars=(i,
                       x_t,
                       given_num,
                       self.gen_x)
        )

        gen_x_2 = self.gen_x.stack()  # seq_length x batch_size
        gen_x_2 = tf.transpose(gen_x_2, perm=[1, 0])
        self.generator.model.reset_states()
        return gen_x_2

    @tf.function
    def get_unrolled_samples(self, given_num, input_x):
        samples = self.autobots(given_num, input_x)
        y_pred_for_auc = self.discriminator.discriminate(samples)
        return y_pred_for_auc

    def get_reward(self, samples):
        """

        :param samples: a init_generated_samples matrix with shape [batch_size, unique_char_size]
        :return: rewards
        """
        rewards = []

        for i in range(self.rollout_num):
            for given_num in tf.range(1, self.seq_length):
                y_pred_for_auc = self.get_unrolled_samples(given_num, samples)
                y_pred = np.array(tf.squeeze(y_pred_for_auc))

                if i == 0:
                    rewards.append(y_pred)
                else:
                    rewards[given_num - 1] += y_pred

            # 最后一个token的reward
            y_pred_for_auc = self.discriminator.discriminate(samples)
            y_pred = np.array(tf.squeeze(y_pred_for_auc))

            if i == 0:
                rewards.append(y_pred)
            else:
                rewards[self.seq_length - 1] += y_pred

        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length

        print()
        print("rollout_get_reward()_rewards.shape:{}".format(rewards.shape))
        print(rewards)

        return rewards



