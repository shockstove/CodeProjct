import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(2114052)


def Generate_Sample(mean, cov, P, label):  # 生成数据集
    # 均值向量 方差矩阵 先验概率
    temp_num = round(1000 * P)
    x, y = np.random.multivariate_normal(mean, cov, temp_num).T
    z = np.ones(temp_num) * label
    X = np.array([x, y, z])
    return X.T


def Generate_Diagram(mean, cov, P):  # 数据集+图表
    dia = []
    label = 1
    for i in range(3):
        dia.append(Generate_Sample(mean[i], cov, P[i], label))
        label += 1
        i += 1

    plt.figure()

    for i in range(3):
        plt.plot(dia[i][:, 0], dia[i][:, 1], '.', markersize=4.)
        plt.plot(mean[i][0], mean[i][1], 'r*')
    plt.show()
    return dia


# 均值向量
mean = np.array([[1, 4], [4, 1], [8, 4]])
# 方差矩阵
cov = [[2, 0], [0, 2]]
# 随机样本个数
num = 500
# X1三个分布模型的先验概率相同
P1 = [1 / 3, 1 / 3, 1 / 3]
# 设置X2先验概率
P2 = [0.6, 0.3, 0.1]
# 生成数据集
X1 = np.array(Generate_Diagram(mean, cov, P1), dtype=object)
X2 = np.array(Generate_Diagram(mean, cov, P2), dtype=object)
X1 = np.vstack(X1)
X2 = np.vstack(X2)
# 数据集分类
X1_1 = X1[np.where(X1[:, 2] == 1.0)]
X1_2 = X1[np.where(X1[:, 2] == 2.0)]
X1_3 = X1[np.where(X1[:, 2] == 3.0)]
X2_1 = X2[np.where(X2[:, 2] == 1.0)]
X2_2 = X2[np.where(X2[:, 2] == 2.0)]
X2_3 = X2[np.where(X2[:, 2] == 3.0)]


# 计算均值和协方差矩阵
def MLE(X):
    mu = np.mean(X, axis=0)
    cov = np.array([np.dot((X[i] - mu).reshape(2, 1), (X[i] - mu).reshape(1, 2)) for i in range(len(X))]).mean(axis=0)
    return mu, cov


# 分别计算三个类别的均值和协方差矩阵
# X1
mu_1_1, cov_1_1 = MLE(X1_1[:, 0:2])
mu_1_2, cov_1_2 = MLE(X1_2[:, 0:2])
mu_1_3, cov_1_3 = MLE(X1_3[:, 0:2])
mean_1 = np.array([list(mu_1_1), list(mu_1_2), list(mu_1_3)])
cov_1 = np.array([list(cov_1_1), list(cov_1_2), list(cov_1_3)])
# X2
mu_2_1, cov_2_1 = MLE(X2_1[:, 0:2])
mu_2_2, cov_2_2 = MLE(X2_2[:, 0:2])
mu_2_3, cov_2_3 = MLE(X2_3[:, 0:2])
mean_2 = np.array([list(mu_2_1), list(mu_2_2), list(mu_2_3)])
cov_2 = np.array([list(cov_2_1), list(cov_2_2), list(cov_2_3)])


def Gaussian(x, mean, cov):
    det_cov = np.linalg.det(cov.tolist())  # 计算方差矩阵的行列式
    inv_cov = np.linalg.inv(cov.tolist())  # 计算方差矩阵的逆
    # 计算概率p(x|w)
    p = 1 / ((2 * np.pi)** (2/2) * np.sqrt(det_cov)) * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean)))
    return p


# 似然率测试规则
def Likelihood_Test(X, mean, cov):
    class_num = mean.shape[0]  # 类的个数
    number = np.array(X).shape[0]  # 样本个数
    wrong = 0  # 分类错误样本数

    for i in range(number):
        p_temp = np.zeros(3)

        for j in range(class_num):
            # 样本i决策到j类的概率
            p_temp[j] = Gaussian(X[i][0:2], mean[j], cov[j])

        p_class = np.argmax(p_temp) + 1  # 样本i决策到的类

        if p_class != X[i][2]:  # 分类错误，增加错误数量
            wrong += 1

    return round(wrong / number, 3)  # 返回误差率


## 最大后验概率规则
def Max_Posterior(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    number = np.array(X).shape[0]  # 样本个数
    wrong = 0  # 分类错误样本数

    for i in range(number):
        p_temp = np.zeros(3)

        for j in range(class_num):
            # 样本i属于j类的后验概率
            p_temp[j] = Gaussian(X[i][0:2], mean[j], cov[j]) * P[j]

        p_class = np.argmax(p_temp) + 1  # 得到样本i所属的类别

        if p_class != X[i][2]:
            wrong += 1  # 分类错误，增加错误数量

    return round(wrong / number, 3)


# 计算似然率测试规则误差
error_Likelihood_Test = Likelihood_Test(X1, mean_1, cov_1)
error_Likelihood_Test_2 = Likelihood_Test(X2, mean_2, cov_2)

# 计算最大后验概率规则误差
error_Max_Posterior = Max_Posterior(X1, mean_1, cov_1, P1)
error_Max_Posterior_2 = Max_Posterior(X2, mean_2, cov_2, P2)

print("对数据集X1，分类错误率如下：")
print("似然率测试规则： \t{}".format(error_Likelihood_Test))
print("最大后验概率规则： \t{}".format(error_Max_Posterior))
print("对数据集X2，分类错误率如下：")
print("似然率测试规则： \t{}".format(error_Likelihood_Test_2))
print("最大后验概率规则： \t{}".format(error_Max_Posterior))


def Gaussian_Kernel(x, X, h=2):
    # 高斯核函数
    # p = (1 / (np.sqrt(2 * np.pi) * h)) * np.array(
    #     [np.exp(-0.5 * np.dot(x - X[i], x - X[i]) / (h ** 2)) for i in range(len(X))]).mean()
    p=0.0
    for i in range(len(X)):
        p += 1.0 / np.sqrt(2 * np.pi * h * h) * np.exp(
            -(np.linalg.norm(x[0:2] - X[i][0:2], ord=2)) ** 2 / (2 * h * h))
    p= p / len(X)
    return p


# 高斯核函数估计方法 + 似然率测试规则
def Gaussian_Likelihood(X, P, h):
    class_num = X.shape[1]  # 类别数目
    num = X.shape[0]  # 样本数目
    wrong = 0  # 错误个数

    for i in range(num):

        class_p = np.zeros(class_num)  # 各类别概率
        start = 0

        for j in range(class_num):
            # 样例属于第j类的概率
            if i >= start and i < start + round(num * P[j]):
                # 删除与测试样例相同的数据点
                data = np.delete(X[start:start + round(num * P[j])], i - start, axis=0)
            else:
                data = X[start:start + round(num * P[j])]
            class_p[j] = Gaussian_Kernel(X[i][0:2], data[:, 0:2], h)

            start += round(num * P[j])

        start_class = np.argmax(class_p) + 1  # 找到概率最大的类别

        if start_class != X[i][2]:  # 如果分类错误，wrong加1
            wrong += 1

    return round(wrong / num, 3)  # 返回错误率


# 高斯核函数估计方法 + 最大后验概率规则
def Gaussian_Posterior(X, P, h):
    class_num = X.shape[1]  # 类别数目
    num = X.shape[0]  # 样本数目
    wrong = 0  # 错误个数

    for i in range(num):
        class_p = np.zeros(class_num)  # 各类别概率
        start = 0

        for j in range(class_num):
            # 样例属于第j类的概率
            if i >= start and i < start + round(num * P[j]):
                data = np.delete(X[start:start + round(num * P[j])], i - start, axis=0)
            else:
                data = X[start:start + round(num * P[j])]
            class_p[j] = Gaussian_Kernel(X[i][0:2], data[:, 0:2], h) * P[j]
            start += round(num * P[j])

        start_class = np.argmax(class_p) + 1  # 找到概率最大的类别

        if start_class != X[i][2]:  # 如果分类错误
            wrong += 1

    return round(wrong / num, 3)


error_gaussian_likelihood = []
error_gaussian_likelihood_2 = []
error_gaussian_posterior = []
error_gaussian_posterior_2 = []

h_total = [0.1, 0.3,0.5, 1, 1.5,1.7, 2, 2.5]
for h in h_total:
    # error_gaussian_likelihood.append(Gaussian_Likelihood(X1, P1, h))
    error_gaussian_likelihood_2.append(Gaussian_Likelihood(X2, P2, h))
    # error_gaussian_posterior.append(Gaussian_Posterior(X1, P1, h))
    error_gaussian_posterior_2.append(Gaussian_Posterior(X2, P2, h))

# print(error_h)
# print("h取值为[0.1, 0.5, 1, 1.5, 2]，X1误差：")
# print("高斯核函数估计方法 + 极大似然规则： \t{}".format(error_gaussian_likelihood))
# plt.figure()
# plt.plot(h_total, error_gaussian_likelihood, 'r')
# plt.grid(linestyle="--")
# plt.xlabel("h")
# plt.ylabel("loss")
# # 显示图
# plt.show()
#
# print("高斯核函数估计方法 + 最大后验规则： \t{}".format(error_gaussian_posterior))
# plt.figure()
# plt.plot(h_total, error_gaussian_posterior, 'r')
# plt.grid(linestyle="--")
# plt.xlabel("h")
# plt.ylabel("loss")
# # 显示图
# plt.show()

print("h取值为[0.1, 0.5, 1, 1.5, 2]，X2误差：")
print("高斯核函数估计方法 + 极大似然规则： \t{}".format(error_gaussian_likelihood_2))
plt.figure()
plt.plot(h_total, error_gaussian_likelihood_2, 'r')
plt.grid(linestyle="--")
plt.xlabel("h")
plt.ylabel("loss")
# 显示图
plt.show()

print("高斯核函数估计方法 + 最大后验规则： \t{}".format(error_gaussian_posterior_2))
plt.figure()
plt.plot(h_total, error_gaussian_posterior_2, 'r')
plt.grid(linestyle="--")
plt.xlabel("h")
plt.ylabel("loss")
# 显示图
plt.show()