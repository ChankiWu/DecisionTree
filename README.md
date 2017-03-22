# DecisionTree
决策树的一个重要任务是为了理解数据中所蕴含的知识信息，因此决策树可以使用不熟悉的数据集合，并从中提取出一系列规则，这些机器根据数据集创建规则的过程，就是机器学习的过程。决策树也是最经常使用的数据挖掘算法。
 3.1 决策树的构造      
 决策树(Decision Tree）是在已知各种情况发生概率的基础上，通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。

决策树的一个重要任务是为了理解数据中所蕴含的知识信息，因此决策树可以使用不熟悉的数据集合，并从中提取出一系列规则，这些机器根据数据集创建规则的过程，就是机器学习的过程。决策树也是最经常使用的数据挖掘算法。

3.1.1 信息增益
划分数据集的大原则是：将无序的数据变得更加有序。再划分数据集前后信息发生的变化称为信息增益（information gain），知道如何计算信息增益，我们就可以计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择。集合信息的度量方式成为香农熵或者简称为熵(entropy)。熵定义为信息的期望值。



计算给定数据集的香农熵

from math import log

def calcShannonEnt(dataSet):
   numEntries = len(dataSet)
   labelCounts = {}
   for featVec in dataSet:
       currentLabel = featVec[-1]
       if currentLabel not in labelCounts.keys():
           labelCounts[currentLabel] = 0
       labelCounts[currentLabel] += 1

   shannonEnt = 0.0
   for key in labelCounts:
       prob = float(labelCounts[key]) / numEntries
       shannonEnt -= prob * log(prob,2)
   return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels



熵越高，则混合的数据也越多。我们可以在数据集中添加更多的分类，观察熵的变化。



基尼不纯度：从一个数据集中随机选取子项，度量其被错误分类到其他分组类的概率。

3.1.2 划分数据集

按照给定特征划分数据集

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
 
代码使用了三个参数：带划分的数据集、划分数据集的特征、需要返回的特征的值。数据集这个列表中的各个元素也是列表，我们要遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中。在if语句中，程序将符合特征的数据抽取出来。







上述代码表示划分出的数据集筛选出第一个元素是1和0的

选择最好的数据集划分方式

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # print numFeatures
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature




函数chooseBestFeatureToSplit（）实现选取特征，划分数据集，计算得出最好的划分数据集的特征。程序的baseEntropy保存了最初的无序度量值，用于与划分完之后的数据集计算的熵值进行比较。第一个for循环遍历数据集中的所有特征，使用列表推导来创建新的列表，将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中，然后使用python'原生的集合（set）数据类型，集合数据类型与列表类型相似，不同之处在于集合类型中的每个值互不相同。然后遍历当前特征中所有的唯一属性值，对每个唯一的属性划分一次数据集，然后计算新的熵值并求和。信息增益是熵的减少或者是数据无序度的减少，最后，比较所有特征中的信息增益，返回最好特征划分的索引值。

代码运行结果告诉我们，第0个特征是最好的用于划分数据集的特征。

3.1.3 递归构建决策树
递归结束的条件是：程序遍历完所有划分数据集的属性、或者每个分支下的所有实例都具有相同的分类。如果所有实例具有相同的分类，则得到一个叶子结点或者终止块。


