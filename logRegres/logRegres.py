# _#_ coding:utf-8 _*_

from numpy import *

# 处理数据
def loadDataSet():
    dataMat= []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():        #逐行读取
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat , labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 原始梯度上升算法
def gradAscent(dataMatIn, classLabels) :
    # 矩阵化
    dataMat = mat(dataMatIn)    # 特征向量
    labelMat = mat(classLabels).transpose()        # 类别标签

    m, n = shape(dataMat)     # 矩阵的大小
    alpha = 0.01    # 移动步长
    maxCycles = 500     #迭代次数
    weights = ones((n,1))     # 初始化回归系数

    for k in range(maxCycles):
        h  = sigmoid(dataMat*weights)
        error = (labelMat - h)         # 计算梯度
        weights = weights + alpha * dataMat.transpose() * error     # 更新回归系数
    return weights

# 原始随机梯度上升算法
# 对比厚厚随机算法：
# 效果相当
# 占用更少的计算资源
# 是一个在线计划
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进随机梯度上升算法
# alpha迭代减小-->缓解高频波动
# 随机更新回归系数 --->减少周期性的波动
# 可以收敛的更快
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))  #initialize to all ones
        for i in range(m):
            alpha = 0.01 + 4 / (1.0 + j + i)     #缓解高频波动
            randIndex = int (random.uniform (0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights)) #随机更新回归系数
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) # 避免重复
    return weights

# 计算Sigmoid值（预测），并取整
def classifyVector(inX, weights):

    prob = sigmoid(sum(inX*weights)) # 是对整向量*回归系数向量 ，然后加和取整

    if prob > 0.5 :
        return 1.0
    else :
        return 0.0

# 处理数据
def colicTest():
    # 训练集导入
    frTrain = open('horseColicTraining.txt')

    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    # 计算回归系数向量
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)

    # 测试集导入并统计错误次数
    frTest = open('horseColicTest.txt')
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))

        # 统计错误次数
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1

    # 计算 错误率
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return(errorRate)

# 重复测试
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))





# 绘图
def plotBestFit(weights):

    import matplotlib.pyplot as plt

    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []
    ycord1 = []

    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
