import tensorflow as tf
from dataLoader import DataLoader, Vocab
from matrix import get_bleu_score
from rollout import Rollout
from generator import Generator

import os
from datetime import datetime
from tqdm import tqdm



def load_data(fasta_file_path: str,
              label_file_path: str,
              shuffle: bool = True,
              seq_length: int = 3000,
              batch_size: int = 64,
              training: bool = True):
    voc = Vocab()
    dataLoader = DataLoader()

    # 全部数据
    dataLoader.sequences = dataLoader.read_fasta_file(fasta_file_path=fasta_file_path)
    # 训练集
    dataLoader.train_seq = dataLoader.sequences[:9000]
    # 测试集
    dataLoader.test_seq = dataLoader.sequences[9000:10000]
    # 标签，0/1
    dataLoader.labels = dataLoader.read_label_txt(label_file_path=label_file_path)
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


# cell_lines = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_line = 'GM12878'
# a = enhancer, promoter
a = 'promoter'

fasta_file_path = fr'E:/Data/EPIs/{cell_line}/train/{cell_line}_{a}.fasta'
label_file_path = fr'E:/Data/EPIs/{cell_line}/train/{cell_line}_label.txt'
batch_size = 37523
embedding_dim = 128  # 256
seq_length = 300
gen_seq_len = 3000
gen_rnn_units = 64  # 1024
learning_rate = 1e-4

# 加载数据
vocab, dataloader = load_data(fasta_file_path=fasta_file_path,
                              label_file_path=label_file_path,
                              shuffle=True,
                              seq_length=seq_length,
                              batch_size=batch_size,
                              training=True)

# 设置路径
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
pretrain_checkpoint_dir = fr'training_checkpoints\{cell_line}\{a}'

# 加载模型
gen = Generator(dataloader=dataloader,
                vocab=vocab,
                batch_size=batch_size,
                embedding_dim=embedding_dim,
                seq_length=seq_length,
                checkpoint_dir=pretrain_checkpoint_dir,
                rnn_units=gen_rnn_units,
                start_token=0,
                learning_rate=learning_rate)

# 加载权重
gen.load_weights()

# 生成数据
# fake_samples = gen.generate(gen_seq_len)
# genned_sentences = vocab.extract_seqs(fake_samples)
# print("生成完毕，写入文件...")
#
# # 保存到本地
# save_fake_samples_filepath = './saved_fake_samples/{}.txt'.format(current_time)
# with open(save_fake_samples_filepath, "w") as f:
#     for genned_sentence in genned_sentences:
#         f.write(genned_sentence + '\n')
# f.close()
# print("写入文件成功！")
