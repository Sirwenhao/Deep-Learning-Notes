# 24/4/18 author:WH

import sys
sys.path.append("..")
import numpy as np
from chap02_001_preprocess import preprocess
from chap02_002_co_matrix import create_co_matrix
from chap02_003_similarity import cos_similarity
from chap02_005_ppmi import ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)

print('covariance matrxi:', C)
print('-'*50)
print('PPMI:', W)

 