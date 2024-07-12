from ucimlrepo import fetch_ucirepo
import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

# metadata
# print(mushroom.metadata)

# variable information
# print(mushroom.variables)

print("X:",X)
print("y:",y)

# Encode labels 'b' as 0 and 'g' as 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Use one-hot encoding for the feature variables X
onehot_encoder = OneHotEncoder(sparse=False)
X_encoded = onehot_encoder.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.020)

# 编译模型
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 创建 EarlyStopping 回调函数
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)

# 训练时间
start_time = time.time()  # 记录开始时间

# 模型训练，并添加 EarlyStopping 回调函数
model.fit([X_train, X_train], y_train, batch_size=32, epochs=1500, validation_data=([X_test, X_test], y_test), callbacks=[early_stopping])
# model.fit([X_train, X_train], y_train, batch_size=32,epochs=60)
# model.fit([X_train, X_train], y_train, batch_size=32, epochs=1500, validation_data=([X_v, X_v], y_v), callbacks=[early_stopping])
# 模型预测
end_time = time.time()  # 记录结束时间
training_time = end_time - start_time  # 计算训练时间（秒）

# 模型预测和测试时间
start_time = time.time()  # 记录开始时间
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