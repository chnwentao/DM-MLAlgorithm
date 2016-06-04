#  Locally weighted linear regression function
#  局部加权线性回归


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


# 计算给定点预测值yHat
def locallyWeightedLR(testPoint,xArr,yArr,k=1.0):
    # 初始化
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]

    # 构建权重对角矩阵
    weights = mat(eye((m))) # 创建对角矩阵，是一个方阵，阶数为样本点个数
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]
        # 权重值以指数级衰减 ，k控制衰减的速度
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) # 高斯核函数

    # 回归系数计算
    xTx = xMat.T * (weights * xMat) # 给每个值赋权
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 局部加权线性回归计算
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    # 对每个点调用locallyWeightedLR
    for i in range(m):
        yHat[i] = locallyWeightedLR(testArr[i],xArr,yArr,k)

    return yHat

# 画图
def lwlrTestPlot(xArr,yArr,k=1.0):
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = locallyWeightedLR(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def test(k = 0.01):
    xArr,yArr=loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr,k)

    xMat=mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])

    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2, c='red')

    plt.show()

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

# 应用app
def application():
    abX,abY= loadDataSet('abalone.txt')

    yHat01=lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1=lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10=lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

    print ("k=0.1 时候的误差:", rssError(abY[0:99],yHat01.T))
    print ("k=1 时候的ri误差:", rssError(abY[0:99],yHat1.T))
    print ("k=10 时候的误差:", rssError(abY[0:99],yHat10.T))
    print ("可以看到较小的核函数 误差小，下面看在新的数据的情况：")

    yHat01=lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    print ("k=0.1 时候的误差:", rssError(abY[100:199],yHat01.T))
    yHat1=lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    print ("k=0.1 时候的误差:", rssError(abY[100:199],yHat1.T))
    yHat10=lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
    print ("k=0.1 时候的误差:", rssError(abY[100:199],yHat10.T))
