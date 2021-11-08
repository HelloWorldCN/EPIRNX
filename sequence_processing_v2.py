import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

cell_lines = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']


def save_to_onehot_from_fasta():
    for cell_line in cell_lines:
        Data_dir = 'E:/Data/EPIs/{}/'.format(cell_line)
        train_dir = Data_dir + 'train/'
        imbltrain = Data_dir + 'imbltrain/'
        test_dir = Data_dir + 'test/'
        print('Experiment on {} dataset'.format(cell_line))

        print('Loading {} seq data...'.format(cell_line))
        enhancers_tra = open(train_dir + '{}_enhancer.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        promoters_tra = open(train_dir + '{}_promoter.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        y_tra = np.loadtxt(train_dir + '{}_label.txt'.format(cell_line))

        im_enhancers_tra = open(imbltrain + '{}_enhancer.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        im_promoters_tra = open(imbltrain + '{}_promoter.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        y_imtra = np.loadtxt(imbltrain + '{}_label.txt'.format(cell_line))

        enhancers_tes = open(test_dir + '{}_enhancer_test.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        promoters_tes = open(test_dir + '{}_promoter_test.fasta'.format(cell_line), 'r').read().splitlines()[1::2]
        y_tes = np.loadtxt(test_dir + '{}_label_test.txt'.format(cell_line))

        print('平衡训练集')
        print('pos_samples:' + str(int(sum(y_tra))))
        print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))
        print('不平衡训练集')
        print('pos_samples:' + str(int(sum(y_imtra))))
        print('neg_samples:' + str(len(y_imtra) - int(sum(y_imtra))))
        print('测试集')
        print('pos_samples:' + str(int(sum(y_tes))))
        print('neg_samples:' + str(len(y_tes) - int(sum(y_tes))))

        encoder = LabelEncoder()
        encoder.fit(['A', 'T', 'C', 'G'])
        enc = OneHotEncoder()

        def get_ont_hot(en_sequences, pr_sequences, ys, EN_MAX_LEN, PR_MAX_LEN):
            en_li = []
            pr_li = []
            y_li = []
            for i in range(len(ys)):
                en_sequence = en_sequences[i]
                pr_sequence = pr_sequences[i]
                y = ys[i]
                if en_sequence.count("N") + pr_sequence.count("N") > 0:
                    continue
                for en_base in en_sequence:
                    en_li.append(en_base)
                for pr_base in pr_sequence:
                    pr_li.append(pr_base)
                y_li.append(y)

            en_enco = enc.fit_transform(np.array([encoder.fit_transform(en_li)]).T).toarray().reshape(-1, EN_MAX_LEN, 4)
            pr_enco = enc.fit_transform(np.array([encoder.fit_transform(pr_li)]).T).toarray().reshape(-1, PR_MAX_LEN, 4)

            print("get_ont_hot中，en_enco.shape={}, pr_enco.shape={}".format(en_enco.shape, pr_enco.shape))
            return en_enco, pr_enco, np.array(y_li)

        enhancers_tra, promoters_tra, y_tra = get_ont_hot(enhancers_tra, promoters_tra, y_tra, 3000, 2000)
        im_enhancers_tra, im_promoters_tra, y_imtra = get_ont_hot(im_enhancers_tra, im_promoters_tra, y_imtra, 3000,
                                                                  2000)
        enhancers_tes, promoters_tes, y_tes = get_ont_hot(enhancers_tes, promoters_tes, y_tes, 3000, 2000)

        print("enhancers_tra.shape={}, promoters_tra.shape={}, y_tra.shape={}".format(enhancers_tra.shape,
                                                                                      promoters_tra.shape, y_tra.shape))
        print("im_enhancers_tra.shape={}, im_promoters_tra.shape={}, y_imtra.shape={}".format(im_enhancers_tra.shape,
                                                                                              im_promoters_tra.shape,
                                                                                              y_imtra.shape))
        print("enhancers_tes.shape={}, promoters_tes.shape={}, y_tes.shape={}".format(enhancers_tes.shape,
                                                                                      promoters_tes.shape, y_tes.shape))

        tra_data_path = Data_dir + cell_line + '_one_hot_tra.npz'
        imtra_data_path = Data_dir + cell_line + '_one_hot_imtrain.npz'
        tes_data_path = Data_dir + cell_line + '_one_hot_tes.npz'

        np.savez(tra_data_path, enhancers_tra=enhancers_tra, promoters_tra=promoters_tra, y_tra=y_tra)
        np.savez(tes_data_path, enhancers_tes=enhancers_tes, promoters_tes=promoters_tes, y_tes=y_tes)
        np.savez(imtra_data_path, im_enhancers_tra=im_enhancers_tra, im_promoters_tra=im_promoters_tra, y_imtra=y_imtra)

        print("Save {} Done!!!".format(cell_line))
        print()


def save_gen_seqs_to_fasta(gen_seqs, fasta_file_path):
    with open(fasta_file_path, mode='a') as f:
        for gen_seq in gen_seqs:
            f.writelines('Augmented Sequence')
            f.writelines(gen_seq)
    f.close()


if __name__ == '__main__':
