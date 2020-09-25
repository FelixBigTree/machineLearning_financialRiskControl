# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:TreeFei
# Create_time:  
# Software: Pycharm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve, roc_curve, roc_auc_score


# 读取数据
train = pd.read_csv(r'D:\Users\Felixteng\Documents\Pycharm Files\loanDefaultForecast\data\train.csv')

testA = pd.read_csv(r'D:\Users\Felixteng\Documents\Pycharm Files\loanDefaultForecast\data\testA.csv')
del testA['n2.2']
del testA['n2.3']


# 分类指标评价

# # 混淆矩阵
'''
若一个实例是正类，并且被预测为正类，即为真正类TP(True Positive )
若一个实例是正类，但是被预测为负类，即为假负类FN(False Negative )
若一个实例是负类，但是被预测为正类，即为假正类FP(False Positive )
若一个实例是负类，并且被预测为负类，即为真负类TN(True Negative )
'''

# ## 给出一组预测值和真实值
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]

# ## 输出混淆矩阵
print('混淆矩阵:\n', confusion_matrix(y_true, y_pred))
'''
混淆矩阵:
 [[1 1]
 [1 1]]

行代表真实，列代表预测
第一行第一列的1，代表真实为第一类且预测成第一类的有1个
第二行第一列的1，代表真实为第二类且预测成第一类的有1个
第一行第二列的1，代表真实为第一类且预测成第二类的有1个
第二行第二列的1，代表真实为第二类且预测成第二类的有1个
'''

# # 准确率 accuracy
'''
准确率是常用的一个评价指标，但是不适合样本不均衡的情况。 
Accuracy = (TP+TN) / (P+N)，即样本中预测正确的百分比，通常来说，正确率越高，分类器越好
'''
print('ACC:', accuracy_score(y_true, y_pred))
'''ACC: 0.5'''

# # 精确率/查准率 precision
'''
precision = TP / (TP + FP)，即样本中判断为正例的样本有多少是真的正例，正确预测为正样本占预测为正样本的百分比
'''
print('Precision:', precision_score(y_true, y_pred))
'''Precision: 0.5'''

# # 召回率/查全率 recall
'''
recall = TP / (TP + FN)，即正样本中有多少判断正确，正确预测为正样本占正样本的百分比
'''
print('Recall:', recall_score(y_true, y_pred))
'''Recall: 0.5'''

# # F1-score 精确率 precision 和召回率 recall 的一种调和平均
'''
f1-score = 2 * (precision * recall) / (precision + recall)，f1分数认为召回率和精准率同等重要

f2-score = (1 + 4) * (precision * recall) / (4 * precision + recall)，f2分数认为召回率的重要程度是精准率的2倍

f0.5-score = (1 + 0.25) * (precision * recall) / (0.25 * precision + recall)，f0.5分数认为召回率的重要程度是精准率的一半

fβ-score = (1 + β²) * (precision * recall) / (β² * precision + recall)
'''
print('F1-score:', f1_score(y_true, y_pred))
'''F1-score: 0.5'''

# # P-R曲线
'''
P-R曲线是描述精确率和召回率变化的曲线
横轴为召回率recall，纵轴为精确率precision

通过置信度对样本进行排序，再逐个样本选择阈值，可以得出每个置信度下的混淆矩阵，再通过计算P和R得到P-R曲线
当一个模型a的P-R曲线完全包住另一个模型b的P-R曲线时，即可认为a优于b。其他情况下，可以使用平衡点，也即F1值，或者曲线下的面积来评估模型的好坏

当召回率不等于0时，P-R曲线的点和ROC曲线的点都能一一对应，因为两条曲线的点都能对应一个置信度阈值确认的混淆矩阵
当一个模型a在P-R上优于b时，a在ROC曲线上也同样会优于b，反过来也同样成立

当正样本个数严重小于负样本个数，数据严重倾斜时，P-R曲线相比较于ROC曲线更加适合
因为当正样本比例减小时，ROC曲线变化不明显，但是P-R曲线的纵坐标，即准确率会出现明显的衰减
因为当样本严重倾斜时，我们假定召回率不变，那么表现较差的模型必然会召回更多的负样本，那么FP(假正例)就会迅速增加，准确率就会大幅衰减

结论：
    一般情况下，模型评估选择P-R或者ROC没啥区别
    但是当正样本的个数严重少于负样本个数时，P-R曲线相比较于ROC曲线能够更加直观的表现模型之间的差异，更加合适
'''
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
plt.plot(recall, precision)

# # 反观ROC曲线
'''
真正类率(True Positive Rate)TPR: TP / (TP + FN),代表分类器预测的正类中实际正实例占所有正实例的比例
纵轴TPR：TPR越大，预测正类中实际正类越多

负正类率(False Positive Rate)FPR: FP / (FP + TN)，代表分类器预测的正类中实际负实例占所有负实例的比例
横轴FPR:FPR越大，预测正类中实际负类越多

理想目标：TPR=1，FPR=0,即roc图中的(0,1)点，故ROC曲线越靠拢(0,1)点，越偏离45度对角线越好，Sensitivity、Specificity越大效果越好
'''
FPR, TPR, thresholds_roc = roc_curve(y_true, y_pred)
plt.plot(FPR, TPR, 'b')

# # AUC
'''
ROC曲线下的面积，越大越好
首先AUC值是一个概率值，随机挑选一个正样本以及负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值
AUC值越大，当前分类算法越有可能将正样本排在负样本前面，从而能够更好地分类。
'''
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC score:', roc_auc_score(y_true, y_scores))
'''AUC score: 0.75'''

# # KS（Kolmogorov-Smirnov）
'''
通过衡量好坏样本累计分布之间的差值，来评估模型的风险区分能力

ks和AUC一样，都是利用TPR、FPR两个指标来评价模型的整体训练效果

不同之处在于，ks取的是TPR和FPR差值的最大值，能够找到一个最优的阈值；AUC只评价了模型的整体训练效果，
并没有指出如何划分类别让预估的效果达到最好，就是没有找到好的切分阈值

与PR曲线相比，AUC和KS受样本不均衡的影响较小，而PR受其影响较大

    ks值<0.2,一般认为模型没有区分能力。
    ks值[0.2,0.3],模型具有一定区分能力，勉强可以接受
    ks值[0.3,0.5],模型具有较强的区分能力。
    ks值大于0.75，往往表示模型有异常。
'''
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
FPR, TPR, thresholds_ks = roc_curve(y_true, y_pred)
KS = abs(FPR - TPR).max()
print('KS值：', KS)
'''KS值： 0.5238095238095237'''


# 简单扫一眼数据
train.info()
train.groupby('isDefault')['id'].count()
'''
可知训练样本数据有80W条，45个特征，大多数特征类型为数值型
标签为字段 'isDefault'
赛题是个二分类问题，同时正负样本比例不均衡，正：负约1：4
有缺失数据需要处理
'''
