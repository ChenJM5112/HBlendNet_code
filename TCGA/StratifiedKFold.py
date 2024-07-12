
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 加载 TSV 格式的数据
data = pd.read_excel("./data/mRMR_400.xlsx")
labels = data.iloc[:, 0].values
features = data.iloc[:, 1:].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# 迭代过程
for train_index, test_index in skf.split(features, labels):
    train_data, test_data = features[train_index], features[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    train_df = pd.DataFrame(train_data, columns=data.columns[1:])
    train_df.insert(0, "Label", train_labels)  # 将标签插入第一列

    test_df = pd.DataFrame(test_data, columns=data.columns[1:])
    test_df.insert(0, "Label", test_labels)  # 将标签插入第一列



    train_file = f"./data/SKFold-data_400/train_data_fold_{fold}.xlsx"
    test_file = f"./data/SKFold-data_400/test_data_fold_{fold}.xlsx"

    train_df.to_excel(train_file, index=False)
    test_df.to_excel(test_file, index=False)
    print(f"第 {fold} 折训练集已保存到 {train_file}")
    print(f"第 {fold} 折测试集已保存到 {test_file}")

    fold += 1
