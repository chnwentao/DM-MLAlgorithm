# _#_ coding:utf-8 _*_

from numpy import *

# 词集模型(无重)
# 统计文档所有的单词并生成无重词汇表)
def createVocabList (dataSet) :
    vocabSet = set([]) # 创建一个空集
    for docment in dataSet: # 遍历文档
        vocabSet = vocabSet | set(docment) #并集（去重）
    return list(vocabSet) #返回无重词汇表，注意：是list，不是set

#  将输入词汇（行）转换为词向量
def setOfWords2Vec (vocabList, inputSet) : # 无重词汇表;文档
    returnVec = [0] * len(vocabList) #创建一个全0向量
    for word in inputSet : # 遍历 inputSet
        if word in vocabList : #有单词出现在 无重词汇表
            returnVec[vocabList.index(word)] = 1 # 对应向量+1
        else : print ("the word: %s is not in my vocabList!" % word)
    return returnVec

# 词袋模型（有重）
def bagOfWord2VecMN (vocabList, inputSet) :
    returnVec = [0]*len(vocabList) #创建一个全0向量
    for word in inputSet: # 遍历 inputSet
        if word in vocabList : #有单词出现在 词袋中
            returnVec[vocabList.index(word)] += 1 #对应向量+1
    return returnVec

# NB训练函数
def trainNB (trainMatrix, trainCategory) : # 文档词向量矩阵；类别标签向量
    numTrainDocs = len (trainMatrix) # 文档矩阵行数（文档数）
    numWords = len(trainMatrix[0]) # 文档矩阵列数（词向量个数）

    # 概率初始化
    # 为防止分类时，有一个概率为0，使乘积为0，所以初始化 概率向量为1 总单词量为2 ；而不是0
    # 传说中的拉普拉斯平滑(Laplace smothing)
    pClass1 = sum (trainCategory) / float (numTrainDocs) # 目标类别文档出现概率
    p0Num = ones(numWords)  # 目标概率向量
    p1Num = ones(numWords) # 非目标概率向量
    p0Denom = 2.0  # 目标文档中文件中总单词量
    p1Denom = 2.0  # 非目标文档中文件中总单词量

    #概率计算
    for i in range(numTrainDocs) :
        if trainCategory[i] == 1 : # 若为目标类别
            p1Num += trainMatrix[i]  # 单词出现次数统计入概率向量0
            p1Denom += sum(trainMatrix[i]) #单词数统计入总单词量
        else : # 若为非目标类别
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #为防止分类相乘 ，过小的数相乘溢出 , 所以在原始概率上求log
    #！！！！ 去对数变负数了！！！但大小关系不变！
    p1Vect = log(p1Num / p1Denom) #计算目标类别中单词出现概率
    p0Vect = log(p0Num / p0Denom) #计算非目标类别中单词出现概率

    return p0Vect, p1Vect, pClass1 #标类别中单词出现概率；非目标类别中单词出现概率；目标类别文档出现概率

# NB分类函数
def classifyNB (vec2Classify, p0Vec, p1Vec, pClass1) : #待分类文档词向量（行）； 训练函数 返回值
    # ！！！！！ 为什么要加上类别的对数概率！！！
    p1 = sum (vec2Classify * p1Vec) + log(pClass1) #因为取了对数，所以概率相乘变为了相加
    p0 = sum (vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0 :
        return 1
    else :
        return 0



################################################

# 小型测试样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 测试函数
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB(array(trainMat),array(listClasses))
    # print (p0V,p1V,pAb )
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

# 原始的NB训练函数（实际不可用）
def trainNB0 (trainMatrix, trainCategory) : # 文档词向量矩阵；类别标签向量
    numTrainDocs = len (trainMatrix) # 文档矩阵行数（文档数）
    numWords = len(trainMatrix[0]) # 文档矩阵列数（词向量个数）
    # 概率初始化
    pClass1 = sum (trainCategory) / float (numTrainDocs) # 目标类别文档出现概率
    p0Num = zeros(numWords)  # 目标概率向量
    p1Num = zeros (numWords) # 非目标概率向量
    p0Denom = 0.0  # 目标文档中文件中总单词量
    p1Denom = 0.0  # 非目标文档中文件中总单词量
    #概率计算
    for i in range(numTrainDocs) :
        if trainCategory[i] == 1 : # 若为目标类别
            p1Num += trainMatrix[i]  # 单词出现次数统计入概率向量0
            p1Denom += sum(trainMatrix[i]) #单词数统计入总单词量
        else :  # 若为非目标类别
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom #计算目标类别中单词出现概率
    p0Vect = p0Num / p0Denom #计算非目标类别中单词出现概率
    return p0Vect, p1Vect, pClass1 #标类别中单词出现概率；非目标类别中单词出现概率；目标类别文档出现概率
