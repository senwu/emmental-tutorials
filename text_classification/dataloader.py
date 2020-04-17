import os
import random
import re

import numpy as np


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_corpus(path, clean=True, TREC=False, encoding="utf8"):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            label, sep, text = line.partition(" ")
            label = int(label)
            text = clean_str(text.strip()) if clean else text.strip()
            labels.append(label)
            data.append(text.split())
    return data, labels


def load_MR(path, seed=1234):
    file_path = os.path.join(path, "rt-polarity.all")
    data, labels = load_corpus(file_path, encoding="latin-1")
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    labels = [labels[i] for i in perm]
    return data, labels


def load_SUBJ(path, seed=1234):
    file_path = os.path.join(path, "subj.all")
    data, labels = load_corpus(file_path, encoding="latin-1")
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    labels = [labels[i] for i in perm]
    return data, labels


def load_CR(path, seed=1234):
    file_path = os.path.join(path, "custrev.all")
    data, labels = load_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    labels = [labels[i] for i in perm]
    return data, labels


def load_MPQA(path, seed=1234):
    file_path = os.path.join(path, "mpqa.all")
    data, labels = load_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [data[i] for i in perm]
    labels = [labels[i] for i in perm]
    return data, labels


def load_TREC(path, seed=1234):
    train_path = os.path.join(path, "TREC.train.all")
    test_path = os.path.join(path, "TREC.test.all")
    train_x, train_y = load_corpus(train_path, TREC=True, encoding="latin-1")
    test_x, test_y = load_corpus(test_path, TREC=True, encoding="latin-1")
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [train_x[i] for i in perm]
    train_y = [train_y[i] for i in perm]
    return train_x, train_y, test_x, test_y


def load_SST(path, seed=1234):
    train_path = os.path.join(path, "stsa.binary.phrases.train")
    valid_path = os.path.join(path, "stsa.binary.dev")
    test_path = os.path.join(path, "stsa.binary.test")
    train_x, train_y = load_corpus(train_path, False)
    valid_x, valid_y = load_corpus(valid_path, False)
    test_x, test_y = load_corpus(test_path, False)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [train_x[i] for i in perm]
    train_y = [train_y[i] for i in perm]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [x for i, x in enumerate(data) if i % nfold != test_id]
    lst_y = [y for i, y in enumerate(labels) if i % nfold != test_id]
    test_x = [x for i, x in enumerate(data) if i % nfold == test_id]
    test_y = [y for i, y in enumerate(labels) if i % nfold == test_id]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x) * 0.9)
    train_x = [lst_x[i] for i in perm[:M]]
    train_y = [lst_y[i] for i in perm[:M]]
    valid_x = [lst_x[i] for i in perm[M:]]
    valid_y = [lst_y[i] for i in perm[M:]]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [x for i, x in enumerate(data) if i % nfold != valid_id]
    train_y = [y for i, y in enumerate(labels) if i % nfold != valid_id]
    valid_x = [x for i, x in enumerate(data) if i % nfold == valid_id]
    valid_y = [y for i, y in enumerate(labels) if i % nfold == valid_id]
    return train_x, train_y, valid_x, valid_y


def load_embedding(path):
    words = []
    vals = []
    with open(path, encoding="utf-8") as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(" ")
                if len(parts) == 2:
                    continue
                words.append(parts[0])
                vals += [float(x) for x in parts[1:]]
    return words, np.asarray(vals).reshape(len(words), -1)
