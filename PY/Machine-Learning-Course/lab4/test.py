# 数据集
# Most Popular Data Set中的wine数据集（对意大利同一地区声场的三种不同品种的酒做大量分析所得出的数据）
# 基本要求
# 采用分层采样的方式将数据集划分为训练集和测试集。
# 给定编写一个朴素贝叶斯分类器，对测试集进行预测，计算分类准确率。
# 中级要求：使用测试集评估模型，得到混淆矩阵，精度，召回率，F值。
# 高级要求：在中级要求的基础上画出三类数据的ROC曲线，并求出AUC值。
# 拓展要求：浅谈ROC曲线和AUC值作为分类评价的合理性。
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# file = open('wine.data', 'r')
data = list(np.loadtxt('wine.data', delimiter=','))
# data = file.readlines()
data = np.array(data)
label = [[], [], []]
data_number = 0
for i in data:
    if i[0] == 1:
        label[0].append(i[1:14])
    elif i[0] == 2:
        label[1].append(i[1:14])
    else:
        label[2].append(i[1:14])
    data_number += 1
train_data = [[], [], []]
test_data = [[], [], []]
data_length = [round(len(label[i]) / 10) for i in range(3)]
num_iterations = 10
error_count = 0
means = [[], [], []]
variances = [[], [], []]
TP = 0
FP = 0
FN = 0
TN = 0
confusion_matrix = np.zeros([3, 3])
record = []

def naive_bayes():
    error_num = 0
    for i in range(3):
        means[i] = np.mean(train_data[i], axis=0)
        variances[i] = np.var(train_data[i], axis=0)
    for i in range(3):
        for j in test_data[i]:
            posterior = []
            for k in range(3):
                likelihood = np.power((j - means[k]), 2) / (2 * variances[k])
                probability = np.log(np.sqrt(2 * np.pi * variances[k])) + likelihood
                probability = np.sum(probability)
                posterior.append(-probability + math.log(len(label[k]) / data_number))
            if np.argmax(posterior) != i:
                error_num += 1
                confusion_matrix[i][np.argmax(posterior)] += 1
            else:
                confusion_matrix[i][i] += 1
            posterior.append(i)
            record.append(posterior)
    return error_num


for i in range(num_iterations):
    train_data = [np.mat(label[j][:i * data_length[j]] + label[j][(i + 1) * data_length[j]:])
                  if (i + 1) * data_length[j] <= len(label[j])
                  else np.mat(label[j][:i * data_length[j]])
                  for j in range(3)]

    test_data = [np.mat(label[j][i * data_length[j]:(i + 1) * data_length[j]])
                 if (i + 1) * data_length[j] <= len(label[j])
                 else np.mat(label[j][i * data_length[j]:])
                 for j in range(3)]
    error_count += naive_bayes()
print(f"分类准确率为：{1 - error_count / data_number}")


precision = []
recall = []
F_measure = []

for i in range(3):
    precision.append(confusion_matrix[i][i] / (confusion_matrix[0][i] + confusion_matrix[1][i] + confusion_matrix[2][i]))
    recall.append(confusion_matrix[i][i] / (confusion_matrix[i][0] + confusion_matrix[i][1] + confusion_matrix[i][2]))
    F_measure.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))

print(precision,recall,F_measure)


def cal_roc():

    for i in range(3):
        fp_rate = []
        tp_rate = []
        auc = 0.0
        sort_data = sorted(record, key = lambda x:x[i])
        pos_record = [j for j in sort_data if j[3] == i]
        neg_record = [j for j in sort_data if j[3] != i]
        pos_number = len(pos_record)
        neg_number = len(neg_record)
        for j in sort_data:
            fp_count = 0
            tp_count = 0
            middle = j[i]
            for k in pos_record:
                if k[i] >= middle:
                    tp_count += 1
            for k in neg_record:
                if k[i]>= middle:
                    fp_count += 1
            fp_rate.append(fp_count / neg_number)
            tp_rate.append(tp_count / pos_number)
        fp_rate.append(0)
        tp_rate.append(0)
        for j in pos_record:
            for k in neg_record:
                if j[i]>k[i]:
                    auc += 1
                elif j[i] == k[i]:
                    auc += 0.5
        auc /= (pos_number * neg_number)
        AUC.append(auc)
        fpr.append(fp_rate)
        tpr.append(tp_rate)
        print(auc)


fpr = []
tpr = []
AUC = []
cal_roc()

plt.figure(figsize=(10,8),dpi=80)
colors = ['r', 'y', 'b']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, AUC[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()