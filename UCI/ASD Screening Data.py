import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# 指定ARFF文件的路径
file_path = './data/Autism-Child-Data.arff'

# 打开ARFF文件并读取数据
with open(file_path, 'r') as arff_file:
    arff_data = arff.load(arff_file)

# 提取标签和特征
data = arff_data['data']
attributes = arff_data['attributes']
# print(attributes)
# 获取标签和特征的名称
attribute_names = [attr[0] for attr in attributes]
label_index = len(attribute_names) - 1

# 分离标签和特征
labels = [row[label_index] for row in data]
features = [row[:label_index] for row in data]
print("标签：", len(labels))
print("特征：", len(features))
print("标签：", labels)
print("特征：", features)
print("特征名称：", attribute_names[:-1])  # 不包括标签列
# 将特征和标签转换为DataFrame

# 'NO' 转换为 0，'YES' 转换为 1
labels = [0 if label == 'NO' else 1 for label in labels]

df = pd.DataFrame(features, columns=attribute_names[:-1])
df['label'] = labels

# 使用独热编码
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df.drop('label', axis=1))

# 分离标签和特征
X = encoded_features
y = df['label']



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# X_train, X_v, y_train, y_v = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 数据预处理：标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义多层感知机（MLP）作为特征提取器
mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3)
])

# 定义输入层
input_a = tf.keras.Input(shape=(X_train.shape[1],))
input_b = tf.keras.Input(shape=(X_train.shape[1],))

# 使用多层感知机提取特征
output_a = mlp(input_a)
output_b = mlp(input_b)

# 添加LSTM层
lstm_units = 128
lstm_dropout = 0.3
lstm = tf.keras.layers.LSTM(units=lstm_units, dropout=lstm_dropout, return_sequences=True)

# 使用LSTM处理序列数据
lstm_output_a = lstm(tf.expand_dims(output_a, axis=1))
lstm_output_b = lstm(tf.expand_dims(output_b, axis=1))

# 比较或组合特征表示
merged = tf.keras.layers.Concatenate()([lstm_output_a[:, -1, :], lstm_output_b[:, -1, :]])

# 添加其他层和输出层
merged = tf.keras.layers.Dense(units=128, activation='relu')(merged)
merged = tf.keras.layers.Dropout(0.3)(merged)
output = tf.keras.layers.Dense(units=1, activation='sigmoid')(merged)

# 创建模型
model = tf.keras.Model(inputs=[input_a, input_b], outputs=[output])

# 定义损失函数和优化器
loss = 'binary_crossentropy'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.035)

# 编译模型
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 创建 EarlyStopping 回调函数
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# 训练时间
start_time = time.time()  # 记录开始时间

# 模型训练，并添加 EarlyStopping 回调函数
model.fit([X_train, X_train], y_train, batch_size=32, epochs=1500, validation_data=([X_test, X_test], y_test),
          callbacks=[early_stopping])
# model.fit([X_train, X_train], y_train, batch_size=32,epochs=30,)
# model.fit([X_train, X_train], y_train, batch_size=32, epochs=1500, validation_data=([X_v, X_v], y_v),
#           callbacks=[early_stopping])


end_time = time.time()  # 记录结束时间
training_time = end_time - start_time  # 计算训练时间（秒）



# 模型预测和测试时间
start_time = time.time()  # 记录开始时间
# 模型预测
y_pred = model.predict([X_test, X_test])
end_time = time.time()  # 记录结束时间
testing_time = (end_time - start_time) * 1000  # 计算测试时间（毫秒）

y_pred_labels = np.round(y_pred)

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)
specificity = recall_score(y_test, y_pred_labels, pos_label=0)
auc1 = roc_auc_score(y_test, y_pred)


# 打印评价指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Specificity:", specificity)
print("AUC:", auc1)
print("训练时间（秒）:", training_time)
print("测试时间（毫秒）:", testing_time)
# import arff
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.svm import SVC
# import xgboost as xgb
# # 指定ARFF文件的路径
# file_path = './data/Autism-Child-Data.arff'
#
# # 打开ARFF文件并读取数据
# with open(file_path, 'r') as arff_file:
#     arff_data = arff.load(arff_file)
#
# # 提取标签和特征
# data = arff_data['data']
# attributes = arff_data['attributes']
# # print(attributes)
# # 获取标签和特征的名称
# attribute_names = [attr[0] for attr in attributes]
# label_index = len(attribute_names) - 1
#
# # 分离标签和特征
# labels = [row[label_index] for row in data]
# features = [row[:label_index] for row in data]
# print("标签：", len(labels))
# print("特征：", len(features))
# print("标签：", labels)
# print("特征：", features)
# print("特征名称：", attribute_names[:-1])  # 不包括标签列
# # 将特征和标签转换为DataFrame
#
# # 'NO' 转换为 0，'YES' 转换为 1
# labels = [0 if label == 'NO' else 1 for label in labels]
#
# df = pd.DataFrame(features, columns=attribute_names[:-1])
# df['label'] = labels
#
# # 使用独热编码
# encoder = OneHotEncoder(sparse=False)
# encoded_features = encoder.fit_transform(df.drop('label', axis=1))
#
# # 分离标签和特征
# X = encoded_features
# y = df['label']
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # 创建随机森林分类器
# # clf = RandomForestClassifier(n_estimators=100, random_state=42)
# # 创建支持向量机分类器
# # clf = SVC(probability=True, random_state=42)
# # 创建XGBoost分类器
# clf = xgb.XGBClassifier(random_state=42)
# # 训练模型
# clf.fit(X_train, y_train)
#
# # 预测
# y_pred = clf.predict(X_test)
#
# # 计算准确率和AUC
# accuracy = accuracy_score(y_test, y_pred)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, y_pred_proba)
#
# print("准确率 (Accuracy):", accuracy)
# print("AUC (Area Under the Curve):", auc)