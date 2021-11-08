import numpy as np
import random


class Vocab:
    def __init__(self):
        self.vocab = sorted(['A', 'T', 'C', 'G'])
        self.vocab_size = len(self.vocab)
        self.char2idx = {k: v for v, k in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.vocab_ids = self.sequence_to_ids(self.vocab).tolist()

        # print("Vocab_char2idx_dict:{}".format(self.char2idx))

    def sequence_to_ids(self, sequence):
        return np.array([self.char2idx[word] for word in sequence])

    def sequences_to_ids(self, sequences):
        ids = []
        for sequence in sequences:
            ids.extend(self.char2idx[word] for word in sequence)
        np_array_ids = np.array(ids)
        return np_array_ids

    def ids_to_sequence(self, ids):
        return np.array([self.idx2char[i] for i in ids])

    def extract_seqs(self, samples):
        """
        传入生成的多组ids
        :param samples: gen出的多组ids
        :return:
        """
        seq_out = []
        for i in range(samples.shape[0]):
            sample_string = ''.join(self.idx2char[np.array(samples[i, ...])])
            seq_out.append(sample_string)

            print("dataLoader_Vocab_extract_seqs()_sample_string: {}".format(sample_string))
        return seq_out



class DataLoader:
    def __init__(self):
        self.Vocab = Vocab()

        self.sequences = []
        self.labels = []
        self.train_seq = []
        self.test_seq = []
        self.train_vectorized_seq = np.array([])
        self.test_vectorized_seq = np.array([])
        self.current_index = 0

    def read_file(self, filePath):
        data = []
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = list(line)
                data.append(line)
        return data

    def read_fasta_file(self, fasta_file_path):
        """
        读取fasta文件
        :param fasta_file_path:
        :return: np array, e.g. sequences[0]=AAAATT...
        """
        self.sequences = open(fasta_file_path, 'r').read().splitlines()[1::2]
        return self.sequences

    def read_label_txt(self, label_file_path):
        """
        读取label文件
        :param label_file_path:
        :return: np array, e.g. labels[0]=1
        """
        self.labels = np.loadtxt(label_file_path)
        return self.labels

    def get_batch(self,
                  shuffle: bool = True,
                  seq_length: int = 3000,
                  batch_size: int = 32,
                  training: bool = True):
        if shuffle:
            if training:
                np.random.shuffle(self.train_vectorized_seq)
            else:
                np.random.shuffle(self.test_vectorized_seq)

        n = self.train_vectorized_seq.shape[0] - 1
        idx = np.random.choice(n - seq_length, batch_size)

        input_batch = [self.train_vectorized_seq[i: i + seq_length] for i in idx]
        output_batch = [self.train_vectorized_seq[i + 1: i + seq_length + 1] for i in idx]

        # print("n=train_vectorized_seq.shape[0]-1={}".format(n))
        # print("np.shape(idx)={}".format(np.shape(idx)))
        # print("np.shape(input_batch)={}".format(np.shape(input_batch)))
        # print("np.shape(output_batch)={}".format(np.shape(output_batch)))

        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch




if __name__ == '__main__':
    voc = Vocab()
    dataLoader = DataLoader()

    dataLoader.sequences = dataLoader.read_fasta_file(filePath='E:/Data/EPIs/GM12878/train/GM12878_enhancer.fasta')
    dataLoader.train_seq = dataLoader.sequences[:900]
    dataLoader.test_seq = dataLoader.sequences[900:1000]
    dataLoader.labels = dataLoader.read_label_txt(filePath='E:/Data/EPIs/GM12878/train/GM12878_label.txt')
    dataLoader.train_vectorized_seq = voc.sequences_to_ids(dataLoader.train_seq)
    dataLoader.test_vectorized_seq = voc.sequences_to_ids(dataLoader.test_seq)

    print(dataLoader.train_vectorized_seq)
    print(dataLoader.test_vectorized_seq)
    x_batch, y_batch = dataLoader.get_batch(shuffle=True, seq_length=500, batch_size=32, training=True)
    print("x_batch.shape={}, y_batch.shape={}".format(x_batch.shape, y_batch.shape))
    print("x_batch[0]:{}".format(x_batch[0]))
    print("y_batch[0]:{}".format(y_batch[0]))