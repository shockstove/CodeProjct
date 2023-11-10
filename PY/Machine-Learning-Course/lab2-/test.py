import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv("winequality-white.csv")

groupData=data.groupby('quality')
for name,group in groupData:
    print(f"{name}:")
    print(group)
numTuples=[20,163,1457,2198,880,175,5]
totalTuples = sum(numTuples)
testRatio=1/5
trainRatio=4/5
trainData=pd.DataFrame()
testData=pd.DataFrame()
for quality in range(3,10):
    qualityData=data[data['quality']==quality]
    trainSub,testSub=train_test_split(qualityData,test_size=testRatio,train_size=trainRatio,stratify=qualityData['quality'])
    trainData=pd.concat([trainData,trainSub])
    testData=pd.concat([testData,testSub])
print("train:",len(trainData))
print("test:",len(testData))

# 中心化代码
def Normalization_fun(x):
    # 特征零均值
    x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
    return x

# 提取特征和标签
X = data.iloc[:, 0:-1]  # N D
X = Normalization_fun(X)
Y = data.iloc[:, -1]

# 可视化中心化后的sulphates特征
import matplotlib.pyplot as plt
plt.hist(X["sulphates"])
plt.show()

# 这里注意一个小trick：回归系数会比特征x多一维，为了向量相乘方便，可以在训练集X左侧添加全为1的一列
data0 = pd.concat([pd.DataFrame(np.ones(X.shape[0]), columns=['x0']), X], axis=1)
data0


x_train=Normalization_fun(trainData.iloc[:, 0:-1])
y_train=Y=trainData.iloc[:, -1]
x_train=np.append(pd.DataFrame(np.ones(x_train.shape[0])),x_train,axis=1)
y_train=list(y_train)
x_train

x_test=Normalization_fun(testData.iloc[:, 0:-1])
y_test=Y=testData.iloc[:, -1]
x_test=np.append(pd.DataFrame(np.ones(x_test.shape[0])),x_test,axis=1)
y_test=list(y_test)
x_test

def calMSE(x,y,theta):
    loss=1/(2*len(x))*np.dot((y-np.dot(x,theta)).T,(y-np.dot(x,theta)))
    return loss


def BGD(x,y,theta,rate,epochs):
    m=len(y)
    cost=np.zeros(epochs)
    for epoch in range(epochs):
        cost[epoch]=calMSE(x,y,theta)
        theta=theta + rate/len(x)*np.dot((y-np.dot(x,theta)).T,x)
    return theta,cost

def SGD(x,y,theta,rate,epochs):
    m = len(y)
    cost = np.zeros(epochs)
    for epoch in range(epochs):
        j=np.random.randint(12)
        cost[epoch] = calMSE(x, y, theta)
        theta = theta + rate * x[j]*(y[j]-np.dot(x[j],theta))
    return theta, cost

def drawMSE(epochs,cost):
    plt.plot(range(1,epochs+1),cost)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.show()


theta = np.zeros(x_train.shape[1]) #初始化回归系数,在梯度下降法中，参数theta的初始值一般都设定为0,回归系数数量=特征数量+1
rate=0.01 #初始化学习率
epochs=2000 #训练次数


#BGD结果
result_train_BGD=BGD(x_train,y_train,theta,rate,epochs)
theta_train_BGD=result_train_BGD[0]
cost_train_BGD=result_train_BGD[1]
cost_test_BGD=calMSE(x_test, y_test, theta_train_BGD)
print("训练集均方误差：",cost_train_BGD[epochs-1])
print("测试集均方误差：",cost_test_BGD)
drawMSE(epochs,cost_train_BGD)

theta = np.zeros(x_train.shape[1]) #初始化回归系数
epochs=2000


#SGD结果
result_train_SGD=SGD(x_train, y_train, theta, rate, epochs)
theta_train_SDG=result_train_SGD[0]
cost_train_SGD=result_train_SGD[1]
cost_test_SGD=calMSE(x_test, y_test, theta_train_SDG)
print("训练集均方误差：", cost_train_SGD[epochs - 1])
print("测试集均方误差：", cost_test_SGD)
drawMSE(epochs, cost_train_SGD)


rates = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
for rate in rates:
    result_train_BGD = BGD(x_train, y_train, theta, rate, epochs)
    theta_train_BGD = result_train_BGD[0]
    cost_train_BGD = result_train_BGD[1]
    cost_test_BGD = calMSE(x_test, y_test, theta_train_BGD)
    plt.plot(range(1, epochs + 1), cost_train_BGD,label=rate)
    print("学习率",rate,"的BGD测试集均方误差为",cost_test_BGD)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.show()

for rate in rates:
    result_train_SGD = SGD(x_train, y_train, theta, rate, epochs)
    theta_train_SGD = result_train_SGD[0]
    cost_train_SGD = result_train_SGD[1]
    cost_test_SGD = calMSE(x_test, y_test, theta_train_SGD)
    plt.plot(range(1, epochs + 1), cost_train_SGD,label=rate)
    print("学习率",rate,"的SGD测试集均方误差为",cost_test_SGD)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.show()

from sklearn.linear_model import Ridge