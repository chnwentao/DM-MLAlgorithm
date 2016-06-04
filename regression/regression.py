
from numpy import *
import matplotlib.pyplot as plt
# 读取数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算最佳拟合曲线
# 利用最小二乘法
def standRegres(xArr,yArr):
    # 训练集和标签矩阵化
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0: # 检测det是否为）
        print ("This matrix is singular, cannot do inverse")
        return
    #最小二乘法计算 回归系数
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 相关性计算
def  correlationCoefficientC( xMat, yMat, ws):
    yHat = xMat*ws
    return corrcoef(yHat.T, yMat)

# 结果输出
def plotrResult():
    x, y =loadDataSet('ex0.txt')
    print ("训练集(前10) ：\n " ,  x[0:10])

    ws = standRegres(x,y)
    print("回归系数 ：\n " ,  ws)


    xMat=mat(x)
    yMat=mat(y)
    yHat = xMat*ws
    print ( "相关系数 ：\n ", correlationCoefficientC( xMat, yMat, ws))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
