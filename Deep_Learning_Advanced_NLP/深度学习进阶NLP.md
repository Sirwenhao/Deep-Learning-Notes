#### 《深度学习进阶自然语言处理》 
24/4/16 author:WH

##### Chapter02关键概念总结

##### 共现矩阵

如此句话“you say goodbye and i say hello.“，

|         | you  | say  | goodbye | and  |  i   | hello |  .   |
| :-----: | :--: | :--: | :-----: | :--: | :--: | :---: | :--: |
|   you   |  0   |  1   |    0    |  0   |  0   |   0   |  0   |
|   say   |  1   |  0   |    1    |  0   |  1   |   1   |  0   |
| goodbye |  0   |  1   |    0    |  1   |  0   |   0   |  0   |
|   and   |  0   |  0   |    1    |  0   |  1   |   0   |  0   |
|    i    |  0   |  1   |    0    |  1   |  0   |   0   |  0   |
|  hello  |  0   |  1   |    0    |  0   |  0   |   0   |  1   |
|    .    |  0   |  0   |    0    |  0   |  0   |   1   |  0   |

##### 余弦相似度

余弦相似度直观的表示了“两个向量在多大程度上指向同一个方向”，$x=(x_1, x_2, x_3,...,x_n)$和$y=(y_1, y_2, y_3,...,y_n)$的两个向量，他们之间的余弦相似度的定义如下：

$$
similarity(x,y)=\frac{x\cdot y}{\Vert x\Vert \Vert y\Vert}=\frac{x_1y_1+x_2y_2+···+·x_ny_n}{\sqrt{x_1^2+···+x_n^2}{\sqrt{y_1^2+···+y_n^2}}}
$$

##### SVD奇异值分解

SVD可以将任意矩阵分解为三个矩阵的乘积：
$$
A=USV^T
$$
其中$S$为除对角线外均为0的对角矩阵。

手动实现矩阵的奇异值分解，矩阵奇异值分解的基本步骤如下：

1. 计算矩阵AAT和ATA的特征值和特征向量
2. 将特征值取平方根得到奇异值
3. 按照奇异值大小排序，得到奇异值矩阵$S$​
4. 利用特征向量构造做奇异向量矩阵$U$和右奇异向量矩阵$V$

```python
# 24/4/18  author:WH

# 矩阵转置

def tranpose(A):
  rows, cols = A.shape
  AT = np.zeros((cols, rows))
  for i in range(rows):
    for j in range(cols):
      AT[i, j] = A[j, i]
  return AT
  

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
```

