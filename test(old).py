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
from sklearn.ensemble import RandomForestClassifier


# 1、对中文进行处理（分词，简单的n-gram,BOW）
# 2、GBDT
# 3、两种错误分析的方法




def sub_classfier(exam_bow_fea_data,exam_bow_fea_target):



    gb = RandomForestClassifier(n_estimators=200)   #TARGET为label2
    print "target:",len(exam_bow_fea_target)
    print "data:",len(exam_bow_fea_data)
    gb.fit(exam_bow_fea_data, exam_bow_fea_target)
    return gb

def findAllTrainning(mainClass,exam_bow_fea):

    resultData = []
    for rec in range(len(exam)):
        if exam.iloc[rec].LABEL1 == mainClass:
            resultData .append( exam_bow_fea[rec] )
    print len(resultData)
    resultTarget = exam[['LABEL2']][exam.LABEL1 == mainClass]
    return resultData,resultTarget


print 'Loading Data'
exam = pd.read_table('train_all.csv',
                     converters={'date': parse},encoding = 'utf-8')


exam_test = pd.read_table('ch2r_test.csv',
                     converters={'date': parse},encoding = 'utf-8')

print len(exam)
print len(exam_test)

#2、分词

exam = exam.drop(['SEGMENT_FULL','SEGMENT_EVERYWORD'],axis=1)
exam_test = exam_test.drop(['SEGMENT_FULL','SEGMENT_EVERYWORD','SEGMENT_OOV','SEGMENT_OOV_EVERYWORD'],axis=1)
exam['SENTENCE'] = [' '.join(jieba.cut(sentence)) for sentence in exam['SENTENCE']]
exam_test['SENTENCE'] = [' '.join(jieba.cut(sentence)) for sentence in exam_test['SENTENCE']]
print exam.head()
exam['SENTENCE'] = exam['SEGMENT'].apply(lambda x:' '.join(x.split('|')))
exam_test['SENTENCE'] = exam_test['SEGMENT'].apply(lambda x:' '.join(x.split('|')))
# 默认是精确模式
# jieba返回一个list
# 用函数去表示一个list

exam.to_csv('Exam_Prep.csv',encoding = 'utf-8')
exam_test.to_csv('Exam_Prep_Test.csv',encoding = 'utf-8')

print "2. CounVector"

print 'CountVect'

vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
exam_bow_fea = vect.fit_transform(exam['SENTENCE']).toarray()
exam_bow_fea_test = vect.transform(exam_test['SENTENCE']).toarray()
print len(vect.get_feature_names())
print ','.join(vect.get_feature_names())
print len(exam_test)
print len(exam_bow_fea_test)



exam_bow_fea_data = exam_bow_fea       #归一化
print len(exam_bow_fea_data)
exam_bow_fea_target = exam['LABEL2']
print len(exam_bow_fea_target)

exam_bow_fea_test_data = exam_bow_fea_test        #归一化
print len(exam_bow_fea_test_data)
exam_bow_fea_test_target = exam_test['LABEL2']
print len(exam_bow_fea_test_target)

#特征读取完毕

print '计算RandomForest分类'


print '16个小类'

print "Training RandomForest"
esti = 400; dep = 7
gb = RandomForestClassifier(n_estimators=200)
gb.fit(exam_bow_fea_data,exam_bow_fea_target)    #直接fit即可，没有明确的标记，不像分类问题
# joblib.dump(gb,"gb.RandomForestClassifierModel")


print sum(exam_bow_fea_test_target == gb.predict(exam_bow_fea_test_data))/1184.0
print sum(exam_bow_fea_test_target == gb.predict(exam_bow_fea_test_data))



print '4个大类：'


exam_bow_fea_target = exam['LABEL1']
exam_bow_fea_test_target = exam_test['LABEL1']

exam_bow_fea_test_result = exam_test['LABEL2']  #终极结果


print "Training RandomForest"

esti = 400; dep = 7
gb = RandomForestClassifier(n_estimators=200)
gb.fit(exam_bow_fea_data,exam_bow_fea_target)    #直接fit即可，没有明确的标记，不像分类问题
# joblib.dump(gb,"gb.RandomForestClassifierModel")

print sum(exam_bow_fea_test_target == gb.predict(exam_bow_fea_test_data))/58.0
print exam_bow_fea_test_target
print gb.predict(exam_bow_fea_test_data)
np.savetxt('1.csv', exam_bow_fea_test_target,fmt='%s', delimiter = '/t')
np.savetxt('2csv', gb.predict(exam_bow_fea_test_data),fmt='%s', delimiter = '/t')



mainClass = [i for i in gb.predict(exam_bow_fea_test_data)]

resultData,resultTarget = findAllTrainning('attitude',exam_bow_fea_data)         #找到其大类的所有小类
gb1 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('shopping',exam_bow_fea_data)         #找到其大类的所有小类
gb2 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('chatting',exam_bow_fea_data)         #找到其大类的所有小类
gb3 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('trouble',exam_bow_fea_data)         #找到其大类的所有小类
gb4 = sub_classfier(resultData,resultTarget)



result = []
for i in range(len(exam_test)):
    print mainClass[i]
    if mainClass[i] == 'attitude':
        result.append( gb1.predict(exam_bow_fea_test_data[i]))
    elif mainClass[i] == 'shopping':
        result.append( gb2.predict(exam_bow_fea_test_data[i]))
    elif mainClass[i] == 'chatting':
        result.append( gb3.predict(exam_bow_fea_test_data[i]))
    elif mainClass[i] == 'trouble':
        result.append( gb4.predict(exam_bow_fea_test_data[i]))


# print sum( result == exam_bow_fea_test_result ) / 58.0
np.savetxt('new.csv', exam_bow_fea_test_result.as_matrix(),fmt='%s', delimiter = '/t')
np.savetxt('re.csv', np.asarray(result).flatten(),fmt='%s', delimiter = '/t')

