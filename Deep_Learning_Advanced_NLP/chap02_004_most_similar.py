# auothor:WH 24/4/15
# 实现查询指定词与其它所有词之间的距离度量及排序

import sys
sys.path.append('..')
import numpy as np

from chap02_001_preprocess import preprocess
from chap02_002_co_matrix import create_co_matrix
from chap02_003_similarity import cos_similarity


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' %query)
        return
    
    print('[query]:' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    print('query_rec:', query_vec)
    
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])
        
    count = 0
    # argsort()是numpy中的一个函数，用于返回数组中元素排序后的索引值
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' %(id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return
        
    
if __name__ == '__main__':
    text = 'You say goodbye I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print('word_to_id:', word_to_id)
    print('id_to_word:', id_to_word)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    print(C)
    
    most_similar('you', word_to_id, id_to_word, C, top=5)
    
        
    
    
    
