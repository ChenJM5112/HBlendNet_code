# mlp lstm
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

from sklearn.metrics.pairwise import cosine_similarity
# 加载Excel数据
# data = pd.read_excel('mRNA+microRNA+DNA甲基化.xlsx')
# data = pd.read_excel('未特征构造_mRNA+microRNA+DNA甲基化.xlsx')
# data = pd.read_excel('压缩感知_mRNA+microRNA+DNA甲基化.xlsx')
import random
# 设置全局随机种子
seed = 47
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
data1 = [


    './data/SKFold-data_400/高斯插值过采样_1.xlsx',
    './data/SKFold-data_400/高斯插值过采样_2.xlsx',
    './data/SKFold-data_400/高斯插值过采样_3.xlsx',
    './data/SKFold-data_400/高斯插值过采样_4.xlsx',
    './data/SKFold-data_400/高斯插值过采样_5.xlsx'

]

data2 = [


    './data/SKFold-data_400/test_data_fold_1.xlsx',
    './data/SKFold-data_400/test_data_fold_2.xlsx',
    './data/SKFold-data_400/test_data_fold_3.xlsx',
    './data/SKFold-data_400/test_data_fold_4.xlsx',
    './data/SKFold-data_400/test_data_fold_5.xlsx'

]


mean_roc_auc = []
mean_accuracy = []
mean_precision = []
mean_recall = []
mean_f1 = []
mean_specificity = []


tpr_list = []
fpr_list = []

def sigmoid_with_noise(x, noise_level=0.03):
    noise = tf.random.uniform(shape=tf.shape(x), minval=-noise_level, maxval=noise_level)
    return 1 / (1 + tf.math.exp(-(x + noise)))
# 在迭代过程之前创建一个空的DataFrame
roc_curve_data = pd.DataFrame(columns=['FPR', 'TPR'])

n = 0
# 迭代过程
for data11, data22 in zip(data1, data2):

    print('正在运行的 训练集 是：', data11)
    print('正在运行的 测试集 是：', data22)
    df1 = pd.read_excel(data11)
    y_train = df1.iloc[:, 0].values
    X_train = df1.iloc[:, 1:].values

    df2 = pd.read_excel(data22)
    y_test = df2.iloc[:, 0].values
    X_test = df2.iloc[:, 1:].values

    print(df1.shape, df2.shape)

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


    # 计算余弦相似度矩阵
    # 使用余弦相似度层连接两个编码
    cosine_similarity = tf.keras.layers.Dot(axes=1, normalize=False)([lstm_output_a[:, -1, :], lstm_output_b[:, -1, :]])


    # 添加其他层和输出层
    merged = tf.keras.layers.Dense(units=128, activation='relu')(cosine_similarity)
    merged = tf.keras.layers.Dropout(0.3)(merged)

    output = tf.keras.layers.Dense(units=1)(merged)
    output = tf.keras.layers.Lambda(sigmoid_with_noise)(output)
    # 创建模型
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=[output])

    # 定义损失函数和优化器
    sigmoid_with_noise = tf.keras.layers.Lambda(sigmoid_with_noise)(output)

    bce_rp_loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.035)

    # 编译模型
    model.compile(optimizer=optimizer, loss=bce_rp_loss, metrics=['accuracy'])

    # 创建 EarlyStopping 回调函数
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)

    # 模型训练，并添加 EarlyStopping 回调函数
    model.fit([X_train, X_train], y_train, batch_size=32, epochs=1500, validation_data=([X_test, X_test], y_test),
              callbacks=[early_stopping])
    # 模型预测
    y_pred = model.predict([X_test, X_test])
    y_pred_labels = np.round(y_pred)

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels)
    specificity = recall_score(y_test, y_pred_labels, pos_label=0)
    auc = roc_auc_score(y_test, y_pred)

    mean_accuracy.append(accuracy)
    mean_precision.append(precision)
    mean_recall.append(recall)
    mean_f1.append(f1)
    mean_specificity.append(specificity)
    mean_roc_auc.append(auc)

    # 打印评价指标
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Specificity:", specificity)
    print("AUC:", auc)
    n=n+1
    print('===============', n, '===============')

    # Calculate ROC curve for the current fold
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    # 将当前迭代的FPR和TPR数据作为新的DataFrame
    iteration_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})

    # 使用concat函数将当前迭代的数据附加到roc_curve_data中
    roc_curve_data = pd.concat([roc_curve_data, iteration_data], ignore_index=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label='Avg. ROC Curve (area = {:.4f})'.format(auc))

    # Add labels and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve')
    plt.legend(loc='lower right')

    # Add a diagonal line for reference
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # Show the plot
    # plt.show()

mean_precision = np.mean(mean_precision)
mean_recall = np.mean(mean_recall)
mean_f1 = np.mean(mean_f1)
mean_specificity = np.mean(mean_specificity)
acc = np.mean(mean_accuracy)
roc_auc = np.mean(mean_roc_auc)

print("平均ACC：", acc)
print("平均precision：", mean_precision)
print("平均Sensitivity-recall：", mean_recall)
print("平均f1：", mean_f1)
print("平均specificity：", mean_specificity)
print("平均AUC：", roc_auc)

# 保存DataFrame到Excel文件
# roc_curve_data.to_excel('new_roc_curve_data.xlsx', index=False)