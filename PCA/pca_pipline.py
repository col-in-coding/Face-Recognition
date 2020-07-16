import numpy as np

X = np.array([
    [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
])

# 标准化
X[0] = X[0] - X[0].mean()
X[1] = X[1] - X[1].mean()

# 计算X协方差矩阵
CX = np.cov(X)

# 降为1维，取最大特征值对应特征向量组成矩阵
w, v = np.linalg.eig(CX)
w_max = w.max()

CY = np.identity(2) * w_max
W = v[:,1]

# 计算投影后的数据
P = W.transpose()
Y = P @ X
print(Y)
