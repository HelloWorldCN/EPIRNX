import random
import numpy as np
import tensorflow as tf
from matrix import get_bleu_score
from dataLoader import DataLoader, Vocab


class GenPretrainer(object):
    def __init__(self, generator, dataloader: DataLoader, vocab: Vocab, pretrain_epochs, tb_writer=None, learning_rate=1e-4):
        self.gen = generator
        self.dataloader = dataloader
        self.pretrain_epochs = pretrain_epochs
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.seq_length = self.gen.seq_length
        self.batch_size = self.gen.batch_size
        self.tb_writer = tb_writer
        self.vocab = vocab

    @tf.function
    def train_step(self, x, y):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            '''TODO: feed the current input into the model and generate predictions'''
            y_hat = self.gen.gen_predictions(x, training=True)

            '''TODO: compute the loss!'''
            loss = self.gen.get_pretrain_loss(labels=y, samples=y_hat)

        # Now, compute the gradients
        '''TODO: complete the function call for gradient computation. 
            Remember that we want the gradient of the loss with respect all 
            of the model parameters. 
            HINT: use `model.trainable_variables` to get a list of all model
            parameters.'''
        grads = tape.gradient(loss, self.gen.model.trainable_variables)

        # Apply the gradients to the optimizer so it can update the model accordingly
        self.optimizer.apply_gradients(zip(grads, self.gen.model.trainable_variables))
        return tf.reduce_mean(loss)

    def pretrain(self, gen_seq_len=None, save_weights=True):

        if gen_seq_len is None:
            gen_seq_len = self.seq_length

        for epoch in range(self.pretrain_epochs):
            # x, y = self.get_pretain_batch()
            x, _ = self.dataloader.get_batch(gen_seq_len, self.batch_size, training=True)
            loss = self.train_step(tf.constant(x), tf.constant(x))

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    tf.summary.scalar('gen_pre_train_loss', loss, step=epoch)
            else:
                print(loss)

            if epoch % 17 == 0 or epoch == 0:
                samples = self.gen.generate(gen_seq_len)
                genned_seqs = self.vocab.sequences_to_ids(samples)
                bleu_score = get_bleu_score(x, genned_seqs)
                print(self.vocab.idx2char[samples[0]])
                if save_weights:
                    self.gen.model.save_weights(self.gen.checkpoint_prefix)
                    # tf.saved_model.save(self.gen, self.gen.checkpoint_prefix)
                gen_loss = self.gen.test_step()
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('gen_pre_test_loss', tf.reduce_mean(gen_loss), step=epoch)
                        tf.summary.scalar('bleu_score', tf.reduce_mean(bleu_score), step=epoch)
