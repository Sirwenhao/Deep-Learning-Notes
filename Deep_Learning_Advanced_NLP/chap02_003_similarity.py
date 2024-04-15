# author:WH 24/4/15
# 向量余弦相似度度量及实现
# 向量余弦相似度：两个向量在多大程度上指向同一个方向

# 余弦相似度定义
# 传入eps防止分母为零

import numpy as np

import sys
sys.path.append('..')
from chap02_001_preprocess import preprocess
from chap02_002_co_matrix import create_co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

if __name__ == '__main__':
    text = 'You Say gooodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)

    c = create_co_matrix(corpus, vocab_size)
    c0 = c[word_to_id['you']]
    c1 = c[word_to_id['i']]
    print(cos_similarity(c0, c1))

