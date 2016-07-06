#coding:utf-8

#coding:utf-8


from dateutil.parser import parse
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.externals import joblib        #用于保存模型
import jieba
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score    #评价标准F值
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# 1、对中文进行处理（分词，简单的n-gram,BOW）
# 2、GBDT
# 3、两种错误分析的方法

print 'Loading Data'
exam = pd.read_table('train_all.csv',
                     converters={'date': parse},encoding = 'utf-8')

exam_test = pd.read_table('ch2r_test.csv',
                     converters={'date': parse},encoding = 'utf-8')

#2、分词

print exam_test
exam = exam.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD'],axis=1)
exam_test = exam_test.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD','SEGMENT_OOV','SEGMENT_OOV_EVERYWORD'],axis=1)
exam['SENTENCE'] = [' '.join(jieba.cut(sentence)) for sentence in exam['SENTENCE']]
exam_test['SENTENCE'] = [' '.join(jieba.cut(sentence)) for sentence in exam_test['SENTENCE']]


# 默认是精确模式
# jieba返回一个list
# 用函数去表示一个list

exam.to_csv('Exam_Prep.csv',encoding = 'utf-8')
exam_test.to_csv('Exam_Prep_Test.csv',encoding = 'utf-8')

print "2. CounVector"

print 'CountVect'

vect = CountVectorizer( ngram_range=(2,2))
exam_bow_fea = vect.fit_transform(exam['SENTENCE']).todense()
exam_bow_fea_test = vect.transform(exam_test['SENTENCE']).todense()

#得到1-gram词袋 化为矩阵形式 方便进行归一化处理
joblib.dump(vect,'exam_vect')      #保存
exam_bow_fea_data = np.log((exam_bow_fea/0.5)+1)        #归一化
print len(exam_bow_fea_data)
exam_bow_fea_target = exam['LABEL']

exam_bow_fea_test_data = np.log((exam_bow_fea_test/0.5)+1)        #归一化
print len(exam_bow_fea_test_data)
exam_bow_fea_test_target = exam_test['LABEL']

#特征读取完毕

print '计算GBDT分类'


print '16个小类'

print "Training GBDT"
esti = 400; dep = 7
gb = GradientBoostingClassifier(n_estimators=esti, max_depth=dep)
gb.fit(exam_bow_fea_data,exam_bow_fea_target)    #直接fit即可，没有明确的标记，不像分类问题
joblib.dump(gb,"gb.GBDTModel")


print "F_socre "+str(f1_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data), average='macro'))
print "Precision " + str(precision_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data),average='macro'))
print "Recall " + str(recall_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data),average='macro'))


print '4个大类：'


exam_bow_fea_target = exam['LABEL1']

exam_bow_fea_test_target = exam_test['LABEL1']



print "Training GBDT"
esti = 400; dep = 7
gb = GradientBoostingClassifier(n_estimators=esti, max_depth=dep)
gb.fit(exam_bow_fea_data,exam_bow_fea_target)    #直接fit即可，没有明确的标记，不像分类问题
joblib.dump(gb,"gb.GBDTModel")


print "F_socre "+str(f1_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data), average='macro'))
print "Precision " + str(precision_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data),average='macro'))
print "Recall " + str(recall_score(exam_bow_fea_test_target, gb.predict(exam_bow_fea_test_data),average='macro'))


