
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# 读取Excel文件
# data = pd.read_excel("./data/SKFold-data_300/train_data_fold_1.xlsx")
# data = pd.read_excel("./data/SKFold-data_300/train_data_fold_2.xlsx")
# data = pd.read_excel("./data/SKFold-data_300/train_data_fold_3.xlsx")
# data = pd.read_excel("./data/SKFold-data_300/train_data_fold_4.xlsx")
data = pd.read_excel("./data/SKFold-data_300/train_data_fold_5.xlsx")

# 将标签列提取出来，并将其转换为标签向量
y = data.iloc[:, 0]

# 将特征列提取出来，并将其转换为特征矩阵
X = data.iloc[:, 1:]



def gaussian_interpolation(x, y, random_state):
    # 高斯插值生成新样本
    mean = (x + y) / 2
    std = abs(y - x) / 2 # 调整方差以控制生成样本的分布范围
    new_sample = random_state.normal(mean, std)
    return new_sample

def generate_synthetic_samples(X, minority_indices, k, N, random_state):
    # 使用K最近邻算法找到每个少数类样本的k个最近邻

    neighbors = NearestNeighbors(n_neighbors=k + 1)  # 加1是因为最近邻中包括自身
    neighbors.fit(X)
    X = X.values
    # print(X[minority_indices])
    _, indices = neighbors.kneighbors(X[minority_indices])

    synthetic_samples = []

    for i, index in enumerate(minority_indices):
        x = X[index]
        k_nearest_indices = indices[i, 1:]  # 排除自身

        for _ in range(N):
            # 随机选择一个最近邻
            random_index = random_state.choice(k_nearest_indices)
            y = X[random_index]

            # 高斯插值生成新样本
            new_sample = gaussian_interpolation(x, y, random_state)
            synthetic_samples.append(new_sample)

    return np.array(synthetic_samples)


def improved_SMOTE(X, y, minority_class, k, N):
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y != minority_class)[0]
    imbalance_ratio = len(majority_indices) // len(minority_indices)

    synthetic_samples = generate_synthetic_samples(X, minority_indices, k, N * imbalance_ratio,
                                                   np.random.RandomState())

    # 合并原始样本和生成的合成样本
    X_resampled = np.vstack((X, synthetic_samples))
    y_resampled = np.hstack((y, np.full(len(synthetic_samples), minority_class)))

    return X_resampled, y_resampled


def auto_detect_minority_class(y):
    counter = Counter(y)
    minority_class = counter.most_common()[-1][0]  # 最少出现的类别
    return minority_class

# 使用改进的SMOTE进行过采样
k = 1
N = 1
minority_class = auto_detect_minority_class(y)
print('少数类是：', minority_class)
os_features, os_labels = improved_SMOTE(X, y, minority_class, k, N)
# 输出增加样本后的数量
print("原始样本数量：", len(X))
print("增加样本后的样本数量：", len(os_features))
# 构建包含标签和特征的DataFrame
synthetic_data = pd.DataFrame(data=os_features, columns=X.columns)
synthetic_data.insert(0, "Label", os_labels)

# 保存为Excel文件
# synthetic_data.to_excel("./data/SKFold-data_400/高斯插值过采样_1.xlsx", index=False)
# synthetic_data.to_excel("./data/SKFold-data_400/高斯插值过采样_2.xlsx", index=False)
# synthetic_data.to_excel("./data/SKFold-data_400/高斯插值过采样_3.xlsx", index=False)
# synthetic_data.to_excel("./data/SKFold-data_400/高斯插值过采样_4.xlsx", index=False)
synthetic_data.to_excel("./data/SKFold-data_400/高斯插值过采样_5.xlsx", index=False)
