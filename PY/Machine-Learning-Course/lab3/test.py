import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# 生成数据集 X1
np.random.seed(15651221)
N = 1000

# 均值向量和协方差矩阵
mu1 = np.array([1, 4])
mu2 = np.array([4, 1])
mu3 = np.array([8, 4])
cov_matrix = np.array([[2, 0], [0, 2]])

# 生成数据点
X1 = np.vstack([
    multivariate_normal.rvs(mu1, cov_matrix, size=int(1 / 3 * N)),
    multivariate_normal.rvs(mu2, cov_matrix, size=int(1 / 3 * N)),
    multivariate_normal.rvs(mu3, cov_matrix, size=int(1 / 3 * N))
])
labels = np.repeat([0, 1, 2], [int(1 / 3 * N), int(1 / 3 * N), int(1 / 3 * N)])
X1 = np.column_stack((X1, labels))
prior_probabilities = [0.6, 0.3, 0.1]
X2 = np.vstack([
    multivariate_normal.rvs(mu1, cov_matrix, size=int(prior_probabilities[0] * N)),
    multivariate_normal.rvs(mu2, cov_matrix, size=int(prior_probabilities[1] * N)),
    multivariate_normal.rvs(mu3, cov_matrix, size=int(prior_probabilities[2] * N))
])
labels = np.repeat([0, 1, 2],
                   [int(prior_probabilities[0] * N), int(prior_probabilities[1] * N), int(prior_probabilities[2] * N)])
X2 = np.column_stack((X2, labels))


def draw(X):
    # 根据标签将数据分类
    class0 = X[X[:, -1] == 0]
    class1 = X[X[:, -1] == 1]
    class2 = X[X[:, -1] == 2]

    # 提取各个类别的x和y坐标
    x0 = class0[:, 0]
    y0 = class0[:, 1]
    x1 = class1[:, 0]
    y1 = class1[:, 1]
    x2 = class2[:, 0]
    y2 = class2[:, 1]

    # 绘制散点图
    plt.scatter(x0, y0, label='Class 0', alpha=0.5)
    plt.scatter(x1, y1, label='Class 1', alpha=0.5)
    plt.scatter(x2, y2, label='Class 2', alpha=0.5)

    # 设置图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Scatter Plot of X')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 显示散点图
    plt.show()


draw(X1)
draw(X2)


# 定义似然率测试规则
def likelihood_ratio_test(x, mu_list, cov_matrix_list, prior_probabilities):
    likelihoods = np.zeros(len(mu_list))
    number = np.array(x).shape[0]
    error_num = 0
    for k in range(number):
        n = 2
        for i in range(len(mu_list)):
            # 计算每个类别的似然函数值
            det = np.linalg.det(cov_matrix_list[i])
            inv_covariance = np.linalg.inv(cov_matrix_list[i])
            exponent = -0.5 * np.dot(np.dot((x[k][0:2] - mu_list[i]).T, inv_covariance), (x[k][0:2] - mu_list[i]))
            coefficient = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
            likelihoods[i] = coefficient * np.exp(exponent)
        if np.argmax(likelihoods) != x[k][2]:
            error_num += 1
    return error_num / number


# 定义最大后验概率测试规则
def max_posterior_test(x, mu_list, cov_matrix_list, prior_probabilities):
    maxs = np.zeros(len(mu_list))
    number = np.array(x).shape[0]
    error_num = 0
    for k in range(number):
        n = len(x[k])
        for i in range(len(mu_list)):
            # 计算每个类别的似然函数值
            det = np.linalg.det(cov_matrix_list[i])
            inv_covariance = np.linalg.inv(cov_matrix_list[i])
            exponent = -0.5 * np.dot(np.dot((x[k][0:2] - mu_list[i]).T, inv_covariance), (x[k][0:2] - mu_list[i]))
            coefficient = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
            maxs[i] = coefficient * np.exp(exponent) * prior_probabilities[i]
        if np.argmax(maxs) != x[k][2]:
            error_num += 1
    return error_num / number


# 应用似然率测试规则
error_rate_x1_likelihood = likelihood_ratio_test(X1, [mu1, mu2, mu3], [cov_matrix, cov_matrix, cov_matrix],
                                                 [1 / 3, 1 / 3, 1 / 3])
error_rate_x1_max = max_posterior_test(X1, [mu1, mu2, mu3], [cov_matrix, cov_matrix, cov_matrix], [1 / 3, 1 / 3, 1 / 3])
error_rate_x2_likelihood = likelihood_ratio_test(X2, [mu1, mu2, mu3], [cov_matrix, cov_matrix, cov_matrix],
                                                 prior_probabilities)
error_rate_x2_max = max_posterior_test(X2, [mu1, mu2, mu3], [cov_matrix, cov_matrix, cov_matrix], prior_probabilities)

print('X1数据集使用极大似然估计方法，错误率为{:.2%}'.format(error_rate_x1_likelihood))
print('X2数据集使用极大似然估计方法，错误率为{:.2%}'.format(error_rate_x2_likelihood))
print('X1数据集使用最大后验估计方法，错误率为{:.2%}'.format(error_rate_x1_max))
print('X2数据集使用最大后验估计方法，错误率为{:.2%}'.format(error_rate_x2_max))

kf = KFold(n_splits=20)

X1_1 = X1[0:333]
X2_1 = X2[0:600]
X1_offset = [333, 666]
X2_offset = [600, 900]


def gaussian_kernel_likelihood_ratio(x, type, x_part_1, mu_list, offset):
    h_value = [0.1, 0.5, 1, 1.5, 2]
    for h in h_value:
        error_rate = 0.0
        for train, test in kf.split(x_part_1):
            X_test = []
            X_train = [[], [], []]
            if type == 1:
                X_test.extend(x[test])
                X_train[0] = (x[train])
                X_test.extend(x[test + offset[0]])
                X_train[1] = (x[train + offset[0]])
                X_test.extend(x[test + offset[1]])
                X_train[2] = (x[train + offset[1]])
            else:
                X_test.extend(x[test])
                X_train[0] = (x[train])
                X_test.extend(x[int(offset[0] + test.min() / 2):int(test.min() / 2 + test.shape[0] / 2 + offset[0])])
                X_train[1] = (x[int(offset[0] + train.min() / 2):int(train.min() / 2 + train.shape[0] / 2 + offset[0])])
                X_test.extend(x[int(offset[1] + test.min() / 6):int(test.min() / 6 + test.shape[0] / 6 + offset[1])])
                X_train[2] = (x[int(offset[1] + train.min() / 6):int(train.min() / 6 + train.shape[0] / 6 + offset[1])])
            number = np.array(X_test).shape[0]
            error_num = 0
            for k in range(number):
                gaussian = np.zeros(len(mu_list))
                for i in range(len(mu_list)):
                    sum = 0.0
                    num = np.array(X_train[i]).shape[0]
                    for j in range(num):
                        sum += 1.0 / np.sqrt(2 * np.pi * h * h) * np.exp(
                            -(np.linalg.norm(X_test[k][0:2] - X_train[i][j][0:2], ord=2) ** 2) / (2 * h * h))
                    gaussian[i] = sum / num
                if np.argmax(gaussian) != X_test[k][2]:
                    error_num += 1
            error_rate += error_num / number
        res = '{:.2%}'.format(error_rate / 20)
        print(f"X{type}数据集使用高斯核函数和极大似然估计方法，在h={round(h, 1)}时分类错误率为{res}")


def gaussian_kernel_max_posterior(x, type, x_part_1, mu_list, offset, prior_probabilities):
    h_value = [0.1, 0.5, 1.0, 1.5, 2.0]
    for h in h_value:
        error_rate = 0.0
        for train, test in kf.split(x_part_1):
            X_test = []
            X_train = [[], [], []]
            if type == 1:
                X_test.extend(x[test])
                X_train[0] = (x[train])
                X_test.extend(x[test + offset[0]])
                X_train[1] = (x[train + offset[0]])
                X_test.extend(x[test + offset[1]])
                X_train[2] = (x[train + offset[1]])
            else:
                X_test.extend(x[test])
                X_train[0] = (x[train])
                X_test.extend(x[int(offset[0] + test.min() / 2):int(test.min() / 2 + test.shape[0] / 2 + offset[0])])
                X_train[1] = (x[int(offset[0] + train.min() / 2):int(train.min() / 2 + train.shape[0] / 2 + offset[0])])
                X_test.extend(x[int(offset[1] + test.min() / 6):int(test.min() / 6 + test.shape[0] / 6 + offset[1])])
                X_train[2] = (x[int(offset[1] + train.min() / 6):int(train.min() / 6 + train.shape[0] / 6 + offset[1])])
            number = np.array(X_test).shape[0]
            error_num = 0
            for k in range(number):
                gaussian = np.zeros(len(mu_list))
                for i in range(len(mu_list)):
                    sum = 0.0
                    num = np.array(X_train[i]).shape[0]
                    for j in range(num):
                        sum += 1.0 / np.sqrt(2 * np.pi * h * h) * np.exp(
                            -(np.linalg.norm(X_test[k][0:2] - X_train[i][j][0:2], ord=2)**2) / (2 * h * h))
                    gaussian[i] = sum * prior_probabilities[i] / num
                if np.argmax(gaussian) != X_test[k][2]:
                    error_num += 1
            error_rate += error_num / number
        res = '{:.2%}'.format(error_rate / 20)
        print(f"X{type}数据集使用高斯核函数和最大后验估计方法，在h={round(h, 1)}时分类错误率为{res}")


gaussian_kernel_likelihood_ratio(X1, 1, X1_1, [mu1, mu2, mu3], X1_offset)
gaussian_kernel_likelihood_ratio(X2, 2, X2_1, [mu1, mu2, mu3], X2_offset)
gaussian_kernel_max_posterior(X1, 1, X1_1, [mu1, mu2, mu3], X1_offset, [1 / 3, 1 / 3, 1 / 3])
gaussian_kernel_max_posterior(X2, 2, X2_1, [mu1, mu2, mu3], X2_offset, prior_probabilities)


# 自定义k-NN函数
def k_nearest_neighbors(X, k):
    num = 320
    x = y = np.linspace(-4, 12, num)
    p = np.zeros((320, 320, 3))
    X_data = X[:, :2]
    for i in range(num):
        for j in range(num):
            point = [x[i], y[j]]
            distances = np.linalg.norm(X_data - point, axis=1)  # 计算欧氏距离
            nearest_indices = np.argsort(distances)[:k]  # 找到最近的k个点的索引
            v = np.pi * (distances[nearest_indices[k - 1]] ** 2)
            class_num = np.zeros(3)
            for t in range(k):
                if X[nearest_indices[t]][2] == 0:
                    class_num[0] += 1
                if X[nearest_indices[t]][2] == 1:
                    class_num[1] += 1
                if X[nearest_indices[t]][2] == 2:
                    class_num[2] += 1
            res = []
            res.append(class_num[0] / (v * N))
            res.append(class_num[1] / (v * N))
            res.append(class_num[2] / (v * N))
            p[i][j] = res
    return p


def knn_draw(X, k):
    p = k_nearest_neighbors(X, k)
    X, Y = np.mgrid[-4:12:320j, -4:12:320j]
    Z0 = p[:, :, 0]
    Z1 = p[:, :, 1]
    Z2 = p[:, :, 2]
    fig = plt.figure(figsize=(12, 4))
    t = plt.subplot(1, 3, 1, projection='3d')
    t.plot_surface(X, Y, Z0, cmap="coolwarm")
    t.set_title("label:0")

    t = plt.subplot(1, 3, 2, projection='3d')
    t.plot_surface(X, Y, Z1, cmap="coolwarm")
    t.set_title("label:0")

    t = plt.subplot(1, 3, 3, projection='3d')
    t.plot_surface(X, Y, Z2, cmap="coolwarm")
    t.set_title("label:0")
    plt.show()


knn_draw(X1,1)
knn_draw(X1,3)
knn_draw(X1,5)
knn_draw(X2,1)
knn_draw(X2,3)
knn_draw(X2,5)
#
# k_value = [1, 3, 5]
# for k in k_value:
#     knn_draw(X1, k)
# k_nearest_neighbors(X1,1)
# # 自定义概率密度估计函数
# def estimate_density(X, x, k):
#     nearest_indices = k_nearest_neighbors(X, x, k)
#     for i in range(k):
#
#     return density
#
# # 数据集
# X1_data = X1[:, :2]
#
# # 选择要估计概率密度的点
# x_to_estimate = np.array([3, 3])
#
# # 对于k=1
# k1_density = estimate_density(X1_data, x_to_estimate, k=1)
#
# # 对于k=3
# k3_density = estimate_density(X1_data, x_to_estimate, k=3)
#
# # 对于k=5
# k5_density = estimate_density(X1_data, x_to_estimate, k=5)
#
# print("Density estimate for k=1:", k1_density)
# print("Density estimate for k=3:", k3_density)
# print("Density estimate for k=5:", k5_density)
