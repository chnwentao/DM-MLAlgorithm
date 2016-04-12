# _#_ coding:utf-8 _*_
#只是一个简化版的以后补充
#
from numpy import *


#简化版SMO
def smoSimple(dataMatIn, classLabels, C, toler, maxIter): #数据集；类别便签；常数；容错率；最大
    # 数据标准化
    dataMatrix = mat(dataMatIn); # 输入数据集矩阵化
    labelMat = mat(classLabels).transpose() # 类别便签转置为列向量
    b = 0;
    m,n = shape(dataMatrix) #行数和列数的tiple
    alphas = mat(zeros((m,1))) #拉个朗日乘子
    #求解并优化alphas
    iter = 0 #保存遍历次数
    while (iter < maxIter):
        alphaPairsChanged = 0 #记录alphas是否优化

        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 预测的类别
            Ei = fXi - float(labelMat[i])# 与真实类别的误差
            # 检查正负边缘（松驰后的） 和 alpha 值
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):

                j = selectJrand(i,m) # 随机第二个alphas
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b #预测类别
                Ej = fXj - float(labelMat[j])  # 误差
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy(); #保存旧alpha值

                #计算 h l 用来保证alpha在0和C中间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print ("L==H");
                    continue

                # 修改alpha [i] & [j]
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T # alpha【j】的最优修改量
                if eta >= 0:
                    print ("eta>=0");
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta #修改alpha[j]
                alphas[j] = clipAlpha(alphas[j],H,L) #保证alpha在0和C中间
                if (abs(alphas[j] - alphaJold) < 0.00001): #alpha[j]有改变就退出循环
                    print ("j not moving enough");
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])# 同时修改 ij

                #设置新的常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                # 记录改变
                alphaPairsChanged += 1 # 记录改变
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        # 统计遍历次数
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

#——————————————————辅助函数——————————————————

# 打开文件并处理
def loadDataSet(filename) :
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines() :
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 在0-m范围内随机一个alpha的下标,但不是i
def selectJrand(i,m) :
    j = i
    while (j == i):
        j = int (random.uniform(0, m)) #随机生成下一个实数，它在[x,y]范围内。
    return j

# 调整alpha的大小
def clipAlpha (aj, H ,L):
    if aj > H:
        aj = H
    if L > aj :
        aj = L
    return aj
