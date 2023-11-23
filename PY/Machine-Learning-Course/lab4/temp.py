import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = open('wine.data', 'r')
types = [[], [], []]                      # 按类分的所有数据
test_data = [[], [], []]
train_data = [[], [], []]
data_num = 0                            # 数据总数
test_len = []                           # 测试集里每一类的个数
means = [[], [], []]                      # 每一类的均值
std = [[], [], []]                        # 每一类的标准差
myline = '1'
while myline:
    myline = f.readline().split(',')
    if len(myline) != 14:
        break
    for t in range(len(myline)):
        if t == 0:
            myline[t] = int(myline[t])
        else:
            myline[t] = float(myline[t])
    temp = myline.pop(0)
    types[temp - 1].append(myline)
test_len = [round(len(types[i]) / 10) for i in range(3)]
data_num = sum([len(types[i]) for i in range(3)])

TP = [0, 0, 0]
FN = [0, 0, 0]
FP = [0, 0, 0]
TN = [0, 0, 0]
precision = [0, 0, 0]
recall = [0, 0, 0]
fp_rate = [[0.0], [0.0], [0.0]]
tp_rate = [[0.0], [0.0], [0.0]]
F_measure = [0, 0, 0]
accuracy = [0, 0, 0]
threshold = 1e-18
p_value = [[], [], []]
label = [[], [], []]
prediction = [[], [], []]

def Bayes(data, p, avg, var):
    result = p
    for i in range(len(data)):
        result *=  1 / (np.sqrt(2 * math.pi)* var[i]) * np.exp(-((data[i] - avg[i]) ** 2) / (2 * var[i]))
    return result


def bayes_classificate():
    # 首先，分别计算训练集上三个类的均值和标准差
    # mean = ...
    # std = ...
    total = 0
    for i in range(3):
        means[i] = np.mean(train_data[i], axis=0)  # 分别计算三个类别的均值
        std[i] = np.std(train_data[i], axis=0)  # 标准差
        total += train_data[i].shape[0]
    wrong_num = 0
    for i in range(3):
        for t in test_data[i]:  # 两层循环：从每一类取每一个测试样本
            my_type = []
            for j in range(3):
                # 由于数据集中所有的属性都是连续值，连续值的似然估计可以按照高斯分布来计算：
                # 这里为了方便计算，将整个式子都取对数处理，连乘变成+，常数项变成log
                temp = np.log((2 * math.pi) ** 0.5 * std[j]) + np.power(t - means[j], 2) / (2 * np.power(std[j], 2))
                temp = np.sum(temp)
                temp = -1 * temp + math.log(len(types[j]) / data_num)
                my_type.append(temp)  # 这里将所有score保存
            pre_type = my_type.index(max(my_type))  # 取分值最大的为预测类别
            p_value[i].append(max(my_type))
            label[i].append(i)

            prediction[i].append(pre_type)
            if math.exp(max(my_type)) > threshold:
                if pre_type == i:
                    # tp_rate[i].append(tp_rate[i][-1] + 1)
                    # fp_rate[i].append(fp_rate[i][-1])
                    TP[i] += 1
                else:
                    # tp_rate[i].append(tp_rate[i][-1])
                    # fp_rate[i].append(fp_rate[i][-1] + 1)
                    FP[i] += 1
            else:
                if pre_type == i:
                    FN[i] += 1
                else:
                    TN[i] += 1
            if pre_type != i:  # 统计错误数
                wrong_num += 1
    return wrong_num


wrong_cnt = [0, 0, 0]
for i in range(10): # 数据集划分9:1，分层抽样
    for j in range(3):
        if (i+1) * test_len[j] > len(types[j]):
            test_data[j] = np.mat(types[j][i * test_len[j]:])
            train_data[j] = np.mat(types[j][:i * test_len[j]])
        else:
            test_data[j] = np.mat(types[j][i * test_len[j]:(i+1) * test_len[j]])
            train_data[j] = np.mat(types[j][:i * test_len[j]]+types[j][(i+1) * test_len[j]:])
    wrong_cnt[j] += bayes_classificate()
print("分类准确率: "+str(1 - sum(wrong_cnt)/data_num))