from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from itertools import permutations
import os

data = list(np.loadtxt('data.dat', delimiter=' '))
data = np.array(data)
labels = list(np.loadtxt('label.dat'))
labels = np.array(labels)

def cal_distance(node1,node2,type):
    # for i in range(len(cluster1)):
    #     node1 = cluster1[i]
    #     for j in range(len(cluster2)):
    #         node2 = cluster2[j]
    # dist = np.sqrt(np.sum(np.square(node1 - node2)))
    dist = np.sum(np.square(node1 - node2))
    # if type == 1:
    #     return np.min(dists)
    # elif type == 2:
    #     return np.max(dists)
    # else:
    #     return np.mean(dists)
    return dist

def modify_distance(matrix,type):
    if type == 1:
        return np.min(matrix,axis=0)
    elif type == 2:
        return np.max(matrix,axis=0)
    else :
        return np.average(matrix,axis=0)


def cal_clustering(type, k):
    clusters = []
    for i in range(data.shape[0]):
        node = [i]
        clusters.append(node)
    dis_matrix = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        # dists = []
        for j in range(i,data.shape[0]):
            if i == j:
                dist = 65536
            else:
                dist = np.sqrt(np.sum(np.square(data[i] - data[j])))
            dis_matrix[i][j] = dist
            dis_matrix[j][i] = dist
    dis_matrix = np.array(dis_matrix)
    while len(clusters) != k:
        min_cluster1 = int(np.argmin(dis_matrix)/dis_matrix.shape[0])
        min_cluster2 = np.argsort(dis_matrix[min_cluster1])[0]
        # for i in clusters[min_cluster2]:
        #     clusters[min_cluster1].append(i)
        clusters[min_cluster1] = clusters[min_cluster1] + clusters[min_cluster2]
        del clusters[min_cluster2]
        temp = modify_distance(dis_matrix[[min_cluster1, min_cluster2]], type)
        dis_matrix[min_cluster1] = temp
        dis_matrix[:, min_cluster1] = temp
        dis_matrix = np.delete(dis_matrix, min_cluster2, axis = 0)
        dis_matrix = np.delete(dis_matrix, min_cluster2, axis = 1)
        if min_cluster1 > min_cluster2:
            min_cluster1 = min_cluster1-1
        dis_matrix[min_cluster1,min_cluster1] = 65536
        # for i in range(len(clusters)):
        #     dis_matrix[min_cluster1, i] = cal_distance(data[clusters[min_cluster1]], data[clusters[i]],type)
        #     dis_matrix[i, min_cluster1] = dis_matrix[min_cluster1,i]
        print("cluster number is ",len(clusters))
    correct_rates = []
    results = []
    cases = [x for x in range(k)]
    cases = permutations(cases)  # 0、1、2、3四种标签的排列情况
    for case in cases:
        temp = np.zeros(labels.shape)
        for i in range(len(case)):
            temp[clusters[i]] = case[i]
        results.append(temp)
        correct_number = 0
        for j in range(temp.shape[0]):
            if temp[j] == labels[j]:
                correct_number += 1
        correct_rate = correct_number / temp.shape[0]
        correct_rates.append(correct_rate)
    correct = np.max(correct_rates)
    result = results[np.argmax(correct_rates)]
    return correct,result


def plot_draw(data, label,type):
    fig = plt.figure()
    t = fig.add_subplot(1,1,1,projection='3d')
    colors = 'rgby'
    for i in range(len(data)):
        t.scatter(data[i][0],data[i][1],data[i][2],color=colors[int(label[i])], s=np.pi)
    plt.title(type)
    plt.show()




correct_single, labels_single = cal_clustering(1, 4)
correct_complete, labels_complete = cal_clustering(2, 4)
correct_average, labels_average = cal_clustering(3, 4)
print("single_linkage层次聚类算法正确率为",correct_single)
plot_draw(data,labels_single,'single_linkage')
print("complete_linkage层次聚类算法正确率为",correct_complete)
plot_draw(data,labels_complete,'complete_linkage')
print("average_linkage层次聚类算法正确率为",correct_average)
plot_draw(data,labels_average,'average_linkage')

