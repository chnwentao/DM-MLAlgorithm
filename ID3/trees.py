# _#_ coding:utf-8 _*_
#
# 修复了书上的一个bug ！
#
# 持久化中使用了最通用的json


from math import log
import operator

# 计算信息熵 （entropy）
def calcShannonEnt(dataSet):
    numErtres = len(dataSet) # 统计行数（实例数量）

    #统计标签出现次数
    labelCounts = {} # 用来保存标签的出现次数的dict ，标签为key
    for featVec in dataSet :
        currentLabel = featVec[-1] # 获得标签
        #！！！这一步很重，对一个没有定义的变量 += 1 ，系统无法接受！！！
        if currentLabel not in labelCounts.keys(): # 如果当前标签没有被统计过
            labelCounts[currentLabel] = 0 # 刚兴建一个标签、
        labelCounts[currentLabel] += 1 #标签出现一次，次数就加一

    #计算熵
    shannonEnt = 0.0
    for key in labelCounts :
        prob = float(labelCounts[key])/numErtres #计算标签出现的概率
        #！看人家怎么求-（A+B+C+++）
        shannonEnt -= prob * log(prob, 2) #以二为底求对数，并求sum的负
    return shannonEnt

# 按给定特征划分数据集
def splitDataSet (dataSet, axis, value) : # 数据集，划分特征，特征的返回值
    retDataSet = [] #为了不修改原始数据，新建一个列表

    #抽签数据
    for featVec in dataSet: #遍历数据集
        if featVec[axis] == value :  #发现符合划分特征就：
            reducedFeatVec = featVec[: axis] #将划分特征前的属性copy出来
            reducedFeatVec.extend(featVec[axis + 1 : ]) # 将划分特征后的属性整体做为一个元素copy
            retDataSet.append(reducedFeatVec) ## 将产生的新数据添加到列表中（！！注意，是不含划分特征）
    return retDataSet

# 选择最佳数据集划分方式 ; 返回的是位置
def chooseBestFeatureToSplit (dataSet) :
    numFeatures = len (dataSet[0]) -1 # 统计有多少个属性(最后一个是标签，不算)
    baseEntropy = calcShannonEnt(dataSet)#计算信息熵
    bestInfoGain = 0.0 # 最好信息增益
    bestFeature = -1 # 初始划分特征
    #计算最佳数据集划分点
    for i in range (numFeatures) : # 遍历数据集中的所有特征值
        #创建分类标签（i列中的所有不同特征值list）
        featList = [exampe[i] for exampe in dataSet]
        uniqueVals = set(featList) # 去重——利用set的唯一性（python中最快的方式）
        newEntropy = 0.0
        # 求i列的信息增益Gain
        for value in uniqueVals : #遍历i列中的所有不同特征值
            subDataSet = splitDataSet(dataSet, i , value) #先按每个同特征值的划分
            prob = len(subDataSet)  / float (len(dataSet)) #计算标签出现的概率
            newEntropy  += prob * calcShannonEnt (subDataSet)  #计算每个划分的熵并累加
        infoGain = baseEntropy - newEntropy #信息增益Gain
        #选择最佳信息增益及对应的划分点
        if (infoGain > bestInfoGain) :
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 子叶主要分类的确定（当决策树所有特征值都使用完时）
def majorityCnt(classList) :
    classCount = {}
    for vote in classList : #统计标签的出现次数
        if vote not in classCount.keys() : # 如果当前标签没有被统计过
            classCount[vote] = 0  # 如果当前标签没有被统计过
        classCount[vote] += 1 #标签出现一次，次数就加一
    #按标签出现次数，从最大到最小排序
    sortedClassCount = sorted (classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0] #返回出现最多的一个标签

# 创建决策树(ID3)
def createTree(dataSet,labels) : # 数据集 ；标签列表
    #出口： 分类的确定
    classList = [example[-1] for example in dataSet] #创建包含所有类标签的list
    if classList.count(classList[0]) == len(classList) : #出口1：所有的标签完全相同
        return classList[0] #那就确定了他的分类了
    if len(dataSet[0]) == 1 : # 出口2：所有特征值类型都使用完了
        return majorityCnt(classList) # 出现最多的分类标签就是他的分类
    #递归
    bestFeat = chooseBestFeatureToSplit(dataSet) # 寻找选择最佳数据集划分的特征值的位置
    bestFeatLabel = labels[bestFeat] # 提取类标签
    myTree = {bestFeatLabel : {}} # 以该类标签为父节点生成树
    labelsCopy = labels[:] # 书上bug , 会在第一次的时候改变原始列表内容
    del(labelsCopy[bestFeat]) # 删除标签列表中的这个类标签
    fealValues = [example[bestFeat] for example in dataSet] #得到列表中包含的所有属性值
    uniqueVals = set(fealValues) # 去重
    for value in uniqueVals : #遍历属性值
        subLabels = labelsCopy[:] #拷贝类标签列表（为了保证每次调用不改变原始列表内容）
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat, value), subLabels)
    return myTree #将返回值插人myTree中
    # 注意myTree的表示方式 ：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

# 使用决策树分类函数
def classify(inputTree, featLabels, testVec ) : # 决策树；标签列表；测试数据集
    firstStr = list(inputTree.keys())[0] # 保存fathernode的标签
    secondDict = inputTree[firstStr] # 保存父节点的sunnode（特征值）
    featIndex = featLabels.index(firstStr)  # 将标签转换为索引
    #遍历整个树
    for key in list(secondDict.keys()) : # 遍历sunnode
        if testVec[featIndex] == key: #类标签匹配
            if type(secondDict[key]).__name__ == 'dict' : #是否达到叶子（叶子是 list /str）
                classLabel = classify (secondDict[key] , featLabels, testVec) #不是就递归
            else : classLabel = secondDict[key] #这就是他的分类
    return classLabel #返回分类

# 持久化决策树
def storeTree (intputTree, filename):
    #import pickle #持久化
    import json
    fw = open(filename, 'w')
    #pickle.dump(intputTree, fw)
    json.dump(intputTree, fw)
    fw.close

# 加载树
def grabTree(filename) :
    #import pickle
    #fr = open(filename,'rb')
    #return pickle.load(fr)
    import json
    fr = open(filename)
    return json.load(fr)

# 简单测试用
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# Test
def TestForLenses (filename) :
    fr = open(filename)
    Lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    LensesLabels = ['age', 'prescript', 'astigmatic' , 'tearRate']
    lenseTree = createTree(Lenses, LensesLabels)
    return lenseTree
