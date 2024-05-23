# 共现矩阵
# 实现基于自然语言的向量化、矩阵化表示

import numpy as np

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            
            # 需要考虑左侧越界情况
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
              
            # 考虑右侧越界情况  
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

if __name__ == "__main__":
    corpus = [0, 1, 2, 3, 4, 1, 5, 6]
    vocab_size = 7
    result = create_co_matrix(corpus, vocab_size)
    print(result)
    
    




