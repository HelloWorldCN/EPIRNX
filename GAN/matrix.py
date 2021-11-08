import nltk


def get_bleu_score(true_seqs, genned_seqs):
    bleu_score = 0.
    true_seqs_list = [list(true_seq) for true_seq in true_seqs]
    for genned_seq in genned_seqs:
        if len(genned_seq) > 0:
            bleu_score += nltk.translate.bleu_score.sentence_bleu(true_seqs_list, list(genned_seq[0]), weights=[1])

    bleu_score /= len(genned_seqs)

    return bleu_score
