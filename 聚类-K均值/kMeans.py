# _#_ coding:utf-8 _*_

from numpy import *

# K-means 算法
def kMeans (dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0] # 数据点数（行数）
    clusterAssment = mat(zeros((m , 2))) #存储每个点的分配结果（簇索引 ，误差）
    centroids = createCent(dataSet, k)  #构建k个质心的集合
    clusterChanged = True #退出迭代标记

    while clusterChanged :
        clusterChanged = False #更新标记

        #更新每个点的簇信息
        #计算每个数据点与质心距离，并根据最小距离更新簇
        for i in range(m): # 遍历数据（）
            minDist = inf #存储点与某个质心的最小距离； is infinity - a value that is greater than any other value.
            minIndex = -1 # 存储质心（他所属的簇）索引
            for j in range(k): # 计算该点与k个质心距离
                distJI = distMeas(centroids[j,:],dataSet[i,:]) #计算第j个质心与该点距离
                if distJI < minDist:  # 如比已知的距离还小
                    minDist = distJI #更新最小距离
                    minIndex = j #更新簇索引
            if clusterAssment[i ,0] != minIndex :  #若簇索引发生变化
                clusterChanged = True  #更新标记
            clusterAssment[i, :] = minIndex ,minDist ** 2 #更新i点的分配结果
            print (centroids)

        #更新质心的位置
        for cent in range(k): # 遍历质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis = 0)  #求平均值（每列的平均值）
    return centroids, clusterAssment #返回质心 ； 分配结果

# 二分K均值法
def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0] # 数据点数（行数）
    clusterAssment = mat(zeros((m , 2))) #存储每个点的分配结果（簇索引 ，误差）
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 计算整个数据集的质心
    centList = [centroid0] #创建一个list保存centroid
    for j in range(m): #遍历数据（）
        clusterAssment[j,1] = distMeas(mat(centroid0), mat(dataSet[j,:])) ** 2 # 计算质心与所有点距离
    #当簇数目小于K,一直迭代
    while (len(centList) < k):
        lowestSSE = inf # 最小sse

        # 遍历每一个簇
        for i in range(len(centList)): #遍历簇列表
            #计算总误差
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0], :] #i簇中的所有点
            #分割簇为2个
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #计算分割后误差
            sseSplit = sum(splitClustAss[:,1]) # 本次划分后的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) # 剩余的数据误差SSE
            # print ('sseSplitm, and notSplit', sseSplit, sseNotSplit)
            # 保存误差最小的那个簇
            if (sseSplit + sseNotSplit) < lowestSSE:  #本次划分的误差如果 低于 划分前
                bestCentToSplit = i #
                bestNewCents = centroidMat # 保护质心
                bestClustAss = splitClustAss.copy() #保存分配结果（簇索引 ，误差）
                lowestSSE = sseSplit + sseNotSplit  # 更新最小sse
        #修改簇索引（根据误差最小的那个簇）
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #修改簇索引
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #修改簇索引
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        #更新质心：replace a centroid with two best centroids
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#
        centList.append(bestNewCents[1,:].tolist()[0])
        # reassign new clusters, and SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment


#——————————————————辅助函数——————————————————
#——————————————————辅助函数——————————————————

# 打开文件并处理
def loadDataSet(filename) :
    dataMat = []
    fr = open(filename)
    for line in fr.readlines() :
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) # 将数据float 化
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧氏距离
def distEclud (vecA, vecB):
    return sqrt (sum (power (vecA -  vecB, 2)))

#构建k个质心的集合
def randCent (dataSet, k):
    n = shape(dataSet)[1] # 记录的属性数
    #！！！易错 ，少括弧
    centroids = mat(zeros((k, n)))
    for j in range(n): #构建k个质心的集合
        minJ = min (dataSet[:,j]) # 第j行最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)# 最大值和最小值的差值
        centroids[:,j] = minJ +rangeJ*random.rand(k,1) #  填充：最小值 + 差值的1~0倍的
    return centroids


#——————————————————————测试部分——————————————
#——————————————————————测试部分——————————————



