# _#_ coding:utf-8 _*_
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#构造分类器
#分类的输入向量 inX 训练集dataSet 向量标签labels K 值
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #读取矩阵的长度(行数)

    #计算欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #计算距离
    sqDiffMat = diffMat**2 #距离的平方
    sqDistances= sqDiffMat.sum(axis = 1) #每行相加
    distances = sqDistances **0.5 ##开方 ，生成距离向量

    # 进行排序（看人家怎么解决排序后和标签对不对的上的问题）
    sortedDistIndicies = distances.argsort() # argsort函数:返回的是索引值

    # 确定前K个元素的主要分类
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] ##返回最小距离的分类
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1 #将分类为索引的值+1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1) ,reverse = True) # py2 中为 iteritems()
    return sortedClassCount[0][0]

#文件处理
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOfLines = fr.readlines()
    numberOfLinse = len(arrayOfLines) #统计文件行数
    returnMat = zeros((numberOfLinse, 3)) # 创建0矩阵（训练集）
    classLabelVector = [] #定义向量标签
    # 解析文件数据到列表
    index = 0
    for line in arrayOfLines:
        line = line.strip() # 删除开头结尾处的空白符
        listFromLine = line.split('\t') # 分割
        returnMat[index,:] = listFromLine[0:3] # 将前三个数存在特征向量中
        # 注意：python默认是字符串，所以要强制
        classLabelVector.append(int(listFromLine[-1])) # 最后一个元素存在classLabelVector中
        index += 1
    return returnMat ,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 计算min 注意，容易少了0
    maxVals = dataSet.max(0) # 查找max
    ranges = maxVals - minVals # 计算 极差
    normDataSet = zeros(shape(dataSet))  # 构造
    m = dataSet.shape[0]  # 返回 行数
    # 生成新的特征值 （n - min）/rang
    normDataSet = dataSet - tile(minVals, (m ,1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 测试代码
def datingClassTest():
    hoRatio = 0.10 # 训练集、检验集 分割比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt') # 读取文件
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化特征值
    m = normMat.shape[0] # MAt行数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m] ,3)
        print ("the classifier came back with: %d, the real answer is: %d \n" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print ("the total error rate is : %f" % (errorCount / float(numTestVecs)))
