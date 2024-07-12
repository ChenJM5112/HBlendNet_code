import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

df = pd.read_excel("./data/多组学_归一化_K近邻填充_删减.xlsx")

# 提取标签列和样本列
label_df = df.iloc[:, 0]  # 第一列为标签
data_df = df.iloc[:, 1:]  # 从第二列开始为样本

X = data_df
# 获取标签列
y = label_df.values.ravel()

print(X.shape)
print(y.shape)

# 使用mRMR算法选择特征
K = 500  # 要选择的特征数量
selected_features = mrmr_classif(X, y, K)


bool_selected_features = np.isin(X.columns, selected_features)

selected_features_df = pd.DataFrame({'feature': X.columns[bool_selected_features]})

X_new = SelectKBest(f_classif, k=len(selected_features)).fit_transform(X[selected_features], y)
print(X_new)

# 网格搜索法选取最佳特征子集
step = 400
search_range = range(step, len(selected_features) + step, step)
param_grid = {"k": search_range}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(SelectKBest(f_classif), param_grid=param_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_new, y)

# 输出最优特征子集和对应的AUC值
best_k = grid_search.best_params_["k"]
best_score = grid_search.best_score_
best_features = selected_features_df.iloc[:best_k, :]['feature'].tolist()
print(f"Best number of features: {best_k}")
print(f"Best AUC: {best_score}")
print(f"Best feature subset: {best_features}")

# 保存最优特征子集
data = X[best_features]
normalized_data_df = data

final_df = pd.DataFrame({'Label': y})

for feature in best_features:
    final_df[feature] = normalized_data_df[feature]

final_df.to_excel('./data/mRMR_400.xlsx', index=False)
print('finish!')