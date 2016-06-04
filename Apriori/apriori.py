# _#_ coding:utf-8 _*_

from numpy import *



# apriori频繁项集产生
# args:
# 	数据集；最小支持度
# ruturn:
# 	频繁项集 ；支持度list
def apriori (dataSet, minSupport = 0.5) :
	C1 = creatC1(dataSet)	# 构建大小为1的候选集
	D = list(map(set, dataSet))		# 将数据集转换为集合列表
	Li, supportData = scanD(D, C1, minSupport) 		# 计算支持度并构建频繁项集
	L = [Li] 	# 保存频繁项集
	k = 2

	while (len(L[k - 2]) > 0):
		Ck = aprioriGen(L[k - 2], k) 	# 构建大小为K的候选集
		Lk , supK = scanD(D, Ck , minSupport) 	# 计算支持度并构建频繁项集
		supportData.update(supK) 	#保存支持度 
		L.append(Lk) 	# 保存频繁项集
		k += 1 
	return L, supportData

# 规则产生
# args ： 频繁项集list, 支持度dict ，最小置信度
def generateRules( L , supportData, minConf = 0.7) :
	bigRuleList = [] # 最终规则列表（含置信度，）
	#规则产生
	for i in range(1,len(L)) : # 遍历 频繁项集list （注意，单个的项集没有规则可言，所以 从1开始）
		for freqSet in L[i] : # 遍历  频繁项集
			# 每个频繁项中可能的规则（右侧）
			H1 = [frozenset([item]) for item in freqSet] 
			# 剪枝
			if (i > 1) :  #  频繁项集的元素个超过2个（产生规则就多于2个）的处理
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf) # 合并 
			else : # 直接计算置信度
				calcConf (freqSet, H1, supportData, bigRuleList, minConf) 
	return bigRuleList


# ------------------------------辅助函数 频繁项集用---------------------------------

# 构建大小为1的候选集
def creatC1(dataSet):
	C1 = []
	for transaction in dataSet: # 遍历数据集
		for item in transaction : # 遍历项集
			if not [item] in C1 : # 查重
				C1.append([item]) # 添加到C1
	C1.sort() # 排序
	## 注意 py2 py3的不同！
	return list (map(frozenset, C1)) # C1中的每个元素都变为不可更改forzenset，（以后可以做我字典使用）

# 计算支持度并产生频繁项集
# args：  候选集合列表；   数据集； 最小支持度
# return :
def scanD(D, Ck, minSupport):
	# 统计：候选集合列表 在 数据集中出现次数 
	ssCnt = {}  #保存计数
	for tid in D: #遍历候选集合列表
		for can in Ck: #遍历数据集
			if can.issubset(tid): # if 候选集合列表 包含在 数据集中
				# 计数 
				if not can in ssCnt: # 数据不在计数字典中
				# if not ssCnt.has_key(can): # py 2.X 中使用
					ssCnt[can] = 1 # 创建项目
				else :
					ssCnt[can] += 1
	# 计算支持度
	numItems = float(len(D)) #候选集合个数
	retList = []
	supportData = {}
	for key in ssCnt :
		support = ssCnt[key] / numItems # 计算支持度
		if support >= minSupport : # 支持度大于最小支持度
			retList.insert(0, key) # 添加到retList 
		supportData[key] = support # 支持度全部添加到 supportData 
	return retList, supportData

# 构建大小为K的候选集 (k-1 的笛卡尔积)
# args： 频繁项集列表； 项集元素个数
# return:
def aprioriGen(Lk, k):

	retList = []
	lenLk = len(Lk) # 计算Lk中的个数
	for i in range(lenLk): # 遍历LK
		for j in range(i+1, lenLk): # 遍历剩下的数
			# 前k-2个值都相同就合并
			L1 = list(Lk[i])[: k - 2]  # 取前k-2个值 减少合并次数
			L2 = list(Lk[j])[: k - 2]
			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j]) #取并集
	return retList


# ------------------------------辅助函数 规则产生用---------------------------------

# 规则评估并剪枝
# 传入： 频繁项集list, 规则， 支持度dict ，规则list ， 最小置信度
def  calcConf(freqSet, H, supportData, br1, minConf):
	prunedH = [] # 保存满足最小最小置信度的规则的列表
	
	for conseq in H :# 遍历H
		# 计算置信度
		conf = supportData[freqSet] / supportData [freqSet - conseq] 
		# 基于置信度剪枝
		if conf >= minConf :
			print(freqSet - conseq, '----> ', conseq, 'conf', conf)
			br1.append((freqSet - conseq, conseq, conf)) 	# 保存规则及置信度
			prunedH.append(conseq) #保存规则右侧
	return prunedH

# 合并并生成规则list
def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7) :
	m = len(H[0]) 	#频繁项产生的右侧规则的元素个数
	if (len(freqSet) > (m + 1)):	# 
		Hmp1 = aprioriGen(H , m  + 1)	#生成无重复组合
		Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf) 	#规则评估并剪枝
		if (len(Hmp1) > 1) : # 如果产生多个右侧规则
			rulesFromConseq (freqSet, Hmp1, supportDat , br1 , minConf) # 迭代 再次评估

# ------------------------------测试用---------------------------------

# test用 简单数据集
def loadDataSet():
	return [[1, 3, 4], [2, 3, 5] , [1, 2, 3, 5] , [2, 5]]









