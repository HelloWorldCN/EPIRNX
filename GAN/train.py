import tensorflow as tf
from dataLoader import DataLoader, Vocab
from matrix import get_bleu_score
from rollout import Rollout
from generator import Generator
from genPretrainer import GenPretrainer
from discriminator import Discriminator
from discPretrainer import DiscPretrainer

from datetime import datetime
from tqdm import tqdm


# # 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_data(filePath: str,
              label_txt_filePath: str,
              shuffle: bool = True,
              seq_length: int = 3000,
              batch_size: int = 64,
              training: bool = True):
    voc = Vocab()
    dataLoader = DataLoader()

    # 全部数据
    dataLoader.sequences = dataLoader.read_fasta_file(fasta_file_path=filePath)
    # 训练集
    dataLoader.train_seq = dataLoader.sequences[:900]
    # 测试集
    dataLoader.test_seq = dataLoader.sequences[900:1000]
    # 标签，0/1
    dataLoader.labels = dataLoader.read_label_txt(label_file_path=label_txt_filePath)
    # 训练集的向量表示
    dataLoader.train_vectorized_seq = voc.sequences_to_ids(dataLoader.train_seq)
    # 测试集的向量表示
    dataLoader.test_vectorized_seq = voc.sequences_to_ids(dataLoader.test_seq)

    # print(dataLoader.train_vectorized_seq)
    # print(dataLoader.test_vectorized_seq)
    # x_batch, y_batch = dataLoader.get_batch(shuffle=shuffle, seq_length=seq_length, batch_size=batch_size, training=training)
    # print("x_batch.shape={}, y_batch.shape={}".format(x_batch.shape, y_batch.shape))
    # print("x_batch[0]:{}".format(x_batch[0]))
    # print("y_batch[0]:{}".format(y_batch[0]))

    return voc, dataLoader


def main(pretrain_checkpoint_dir,
         train_summary_writer,
         vocab: Vocab,
         dataloader: DataLoader,
         batch_size: int = 64,
         embedding_dim: int = 256,
         seq_length: int = 3000,
         gen_seq_len: int = 3000,
         gen_rnn_units: int = 1024,
         disc_rnn_units: int = 1024,
         epochs: int = 40000,
         pretrain_epochs: int = 4000,
         learning_rate: float = 1e-4,
         rollout_num: int = 2,
         gen_pretrain: bool = False,
         disc_pretrain: bool = False,
         load_gen_weights: bool = False,
         load_disc_weights: bool = False,
         save_gen_weights: bool = True,
         save_disc_weights: bool = True,
         disc_steps: int = 3
         ):
    gen = Generator(dataloader=dataloader,
                    vocab=vocab,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    seq_length=seq_length,
                    checkpoint_dir=pretrain_checkpoint_dir,
                    rnn_units=gen_rnn_units,
                    start_token=0,
                    learning_rate=learning_rate)
    if load_gen_weights:
        gen.load_weights()
    if gen_pretrain:
        gen_pre_trainer = GenPretrainer(gen,
                                        dataloader=dataloader,
                                        vocab=vocab,
                                        pretrain_epochs=pretrain_epochs,
                                        tb_writer=train_summary_writer,
                                        learning_rate=learning_rate)
        print('Start pre-training generator...')
        gen_pre_trainer.pretrain(gen_seq_len=gen_seq_len, save_weights=save_gen_weights)

    disc = Discriminator(vocab_size=vocab.vocab_size,
                         embedding_dim=embedding_dim,
                         rnn_units=disc_rnn_units,
                         batch_size=batch_size,
                         checkpoint_dir=pretrain_checkpoint_dir,
                         learning_rate=learning_rate)
    if load_disc_weights:
        disc.load_weights()
    if disc_pretrain:
        disc_pre_trainer = DiscPretrainer(disc,
                                          gen,
                                          dataloader=dataloader,
                                          vocab=vocab,
                                          pretrain_epochs=pretrain_epochs,
                                          tb_writer=train_summary_writer,
                                          learning_rate=learning_rate)
        print('Start pre-training discriminator...')
        disc_pre_trainer.pretrain(save_disc_weights)
    rollout = Rollout(generator=gen,
                      discriminator=disc,
                      vocab=vocab,
                      batch_size=batch_size,
                      seq_length=seq_length,
                      rollout_num=rollout_num)

    with tqdm(desc='Epoch: ', total=epochs, dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            fake_samples = gen.generate()
            rewards = rollout.get_reward(samples=fake_samples)
            gen_loss = gen.train_step(fake_samples, rewards)
            real_samples, _ = dataloader.get_batch(shuffle=shuffle,
                                                   seq_length=seq_length,
                                                   batch_size=batch_size,
                                                   training=True)
            disc_loss = 0
            for i in range(disc_steps):
                disc_loss += disc.train_step(fake_samples, real_samples) / disc_steps

            with train_summary_writer.as_default():
                tf.summary.scalar('gen_train_loss', gen_loss, step=epoch)
                tf.summary.scalar('disc_train_loss', disc_loss, step=epoch)
                tf.summary.scalar('total_train_loss', disc_loss + gen_loss, step=epoch)

            pbar.set_postfix(gen_train_loss=tf.reduce_mean(gen_loss),
                             disc_train_loss=tf.reduce_mean(disc_loss),
                             total_train_loss=tf.reduce_mean(gen_loss + disc_loss))

            if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
                print('保存weights...')
                # 保存weights
                gen.model.save_weights(gen.checkpoint_prefix)
                disc.model.save_weights(disc.checkpoint_prefix)
                # gen.model.save('gen.h5')
                # disc.model.save('disc.h5')

                # 测试 disc
                fake_samples = gen.generate(gen_seq_len)
                real_samples = dataloader.get_batch(shuffle=shuffle,
                                                    seq_length=gen_seq_len,
                                                    batch_size=batch_size,
                                                    training=False)
                disc_loss = disc.test_step(fake_samples, real_samples)

                # 测试 gen
                gen_loss = gen.test_step()

                # 得到bleu_score
                # bleu_score = get_bleu_score(true_seqs=real_samples, genned_seqs=fake_samples)
                genned_sentences = vocab.extract_seqs(fake_samples)
                # print(genned_sentences)
                # print(vocab.idx2char[fake_samples[0]])

                # 记录 test losses
                with train_summary_writer.as_default():
                    tf.summary.scalar('disc_test_loss', tf.reduce_mean(disc_loss), step=epoch)
                    tf.summary.scalar('gen_test_loss', tf.reduce_mean(gen_loss), step=epoch)
                    # tf.summary.scalar('bleu_score', tf.reduce_mean(bleu_score), step=epoch + gen_pretrain * pretrain_epochs)

            pbar.update()


if __name__ == '__main__':
    # GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK
    cell_line = 'IMR90'
    # a = promoter, enhancer
    a = 'promoter'
    filePath = 'E:/Data/EPIs/{}/imbltrain/{}_{}.fasta'.format(cell_line, cell_line, a)
    label_txt_filePath = 'E:/Data/EPIs/{}/imbltrain/{}_label.txt'.format(cell_line, cell_line)
    batch_size = 16
    embedding_dim = 128  # 256
    seq_length = 300
    gen_seq_len = 3000
    gen_rnn_units = 64  # 1024
    disc_rnn_units = 64  # 1024
    epochs = 20
    pretrain_epochs = 3
    learning_rate = 1e-4
    rollout_num = 3
    gen_pretrain = False
    disc_pretrain = False
    load_gen_weights = False
    load_disc_weights = False
    save_gen_weights = True
    save_disc_weights = True
    disc_steps = 3
    shuffle = True  # 训练数据是否打乱

    # 加载数据
    voc, dataloader = load_data(filePath=filePath,
                                label_txt_filePath=label_txt_filePath,
                                shuffle=True,
                                seq_length=seq_length,
                                batch_size=batch_size,
                                training=True)

    # 设置路径
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/' + current_time + '/train/{}'.format(cell_line)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    pretrain_checkpoint_dir = './training_checkpoints/{}/{}'.format(cell_line, a)

    main(pretrain_checkpoint_dir,
         train_summary_writer,
         vocab=voc,
         dataloader=dataloader,
         batch_size=batch_size,
         embedding_dim=embedding_dim,
         seq_length=seq_length,
         gen_seq_len=gen_seq_len,
         gen_rnn_units=gen_rnn_units,
         disc_rnn_units=disc_rnn_units,
         epochs=epochs,
         pretrain_epochs=pretrain_epochs,
         learning_rate=learning_rate,
         rollout_num=rollout_num,
         gen_pretrain=gen_pretrain,
         disc_pretrain=disc_pretrain,
         load_gen_weights=load_gen_weights,
         load_disc_weights=load_disc_weights,
         save_gen_weights=save_gen_weights,
         save_disc_weights=save_disc_weights,
         disc_steps=disc_steps)
