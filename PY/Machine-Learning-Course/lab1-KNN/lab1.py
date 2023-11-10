import numpy as np
from sklearn.model_selection import LeaveOneOut, train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from keras import layers, models
import cv2

def ImgtoMat(filename):
    f = open(filename)
    ss = f.readlines()
    l = len(ss)
    f.close()
    returnMat = np.zeros((l, 256))
    returnClassVector = np.zeros((l, 1))
    for i in range(l):
        s1 = ss[i].split()
        for j in range(256):
            returnMat[i][j] = np.float64(s1[j])
        clCount = 0
        for j in range(256, 266):
            if s1[j] == '1':
                clCount = j - 256
                break
        returnClassVector[i] = clCount
    return returnMat, returnClassVector


X, y = ImgtoMat('semeion.data')
np.shape(X), np.shape(y)


def knn(X, Y, neighbors):
    loo = LeaveOneOut()
    testRes = []
    corCount = 0
    for trainIndex, testIndex in loo.split(X, Y):
        xTrain, xTest, yTrain, yTest = X[trainIndex], X[testIndex], Y[trainIndex], Y[testIndex]
        trainVol = xTrain.shape[0]
        testVol = xTest.shape[0]
        diffMat = np.tile(xTest[0], (trainVol, 1)) - xTrain
        sqDiffMat = diffMat ** 2
        sqDistance = sqDiffMat.sum(axis=1)
        distance = sqDistance ** 0.5
        sortDistance = np.argsort(distance)
        labelCount = []
        for j in range(neighbors):  # 考察k近邻属于哪些类
            labelCount.append(yTrain[sortDistance[j]][0])
        classifyRes = Counter(labelCount)  # 把k近邻中最多的那个标签作为分类结果
        classifyRes = classifyRes.most_common(2)[0][0]
        testRes.append(classifyRes)
        if yTest[0] == classifyRes:
            corCount += 1
    corRate = corCount / 1593
    print('k={0}时，测试个数为1593  正确个数为：{1}  准确率为：{2}'.format(neighbors, corCount, corRate))
    return corRate


knn(X, y, 1)
knn(X, y, 3)
knn(X, y, 5)


def sklearnKnn(X, Y, neighbors):
    loo = LeaveOneOut()
    testRes = []
    corCount = 0
    for trainIndex, testIndex in loo.split(X, Y):
        xTrain, xTest, yTrain, yTest = X[trainIndex], X[testIndex], Y[trainIndex], Y[testIndex]
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        neigh.fit(xTrain, yTrain.ravel())
        predict_y = neigh.predict(xTest)
        if yTest[0] == predict_y:
            corCount += 1
    corRate = corCount / Y.shape[0]
    print('k={0}时，测试个数为{1}  正确个数为：{2}  准确率为：{3}'.format(neighbors, Y.shape[0], corCount, corRate))
    return corRate


sklearnKnn(X, y, 1)
sklearnKnn(X, y, 3)
sklearnKnn(X, y, 5)

model = models.Sequential([
    layers.Reshape((16, 16, 1), input_shape=(256,)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def cnn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    augmented_X_train = []
    augmented_y_train = []
    for i in range(len(X_train)):
        image = X_train[i].reshape(16, 16)  # 将扁平的图像数据重塑为16x16
        for angle in [-10, 0, 10]:  # 旋转角度范围
            rotated_image = rotate_image(image, angle)
            augmented_X_train.append(rotated_image.flatten())  # 重塑为扁平的数据
            augmented_y_train.append(y_train[i])
    X_train = np.vstack([X_train, np.array(augmented_X_train)])
    y_train = np.concatenate([y_train, np.array(augmented_y_train)])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
cnn (X, y)