import numpy as np


def tranpose(A):
  rows, cols = A.shape
  AT = np.zeros((cols, rows))
  for i in range(cols):
    for j in range(rows):
      AT[i, j] = A[j, i]
  return AT


def gen_ortho_matrix(n):
    """
    生成单位正交矩阵
    """
    if n <= 0:
        raise ValueError("n 必须大于 0")
    Q = np.zeros((n, n))
    for i in range(n):
        Q[:, i] = np.random.randn(n)
    Q = orthonormalize(Q)
    return Q



def eig(A):
    """
    求矩阵的特征值和向量
    """
    n = A.shape[0]
    if n <= 1:
        return A[0, 0], np.array([1])
    iter_max = 100
    eps = 1e-8
    
    Q = gen_ortho_matrix(n)
    
import numpy as np
def svd(A, tol=1e-10):
    """
    奇异值分解
    """
    AAT = np.dot(A, A.T)
    ATA = np.dot(A.T, A)
    
    # 计算ATA和AAT的特征值和特征向量
    _, U = np.linalg.eig(ATA)
    _, V = np.linalg.rig(AAT)
    
    # 计算奇异值
    s = np.sqrt(np.sort(np.linalg.eigvals(ATA))[::-1])
    
    # 截断奇异值
    rank = np.sum(s > tol)
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]
    
    # 计算右奇异矩阵的转置
    Vt = V.T
    
    return U, s, Vt
    


if __name__ == "__main__":
    A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
    
    A = np.array([
    [1, 2],
    [4, 5],
    [7, 8],
])
    print(tranpose(A))