#coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from dateutil.parser import parse
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

# 1、对中文进行处理（分词，简单的n-gram,BOW）
# 2、GBDT
# 3、两种错误分析的方法


f = open("result.txt","w")
print 'Loading Data'
exam = pd.read_table('train_all.csv',
                     converters={'date': parse},encoding = 'utf-8')


exam_test = pd.read_table('ch2r_test.csv',
                     converters={'date': parse},encoding = 'utf-8')

print len(exam)
print len(exam_test)

#2、分词

exam = exam.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD'],axis=1)
exam_test = exam_test.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD','SEGMENT_OOV','SEGMENT_OOV_EVERYWORD'],axis=1)
exam['SENTENCE'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in exam['SENTENCE']]
exam_test['SENTENCE'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in exam_test['SENTENCE']]

# 默认是精确模式
# jieba返回一个list
# 用函数去表示一个list

exam.to_csv('Exam_Prep.csv',encoding = 'utf-8')
exam_test.to_csv('Exam_Prep_Test.csv',encoding = 'utf-8')

print "2. TFIDF"

vect = TfidfVectorizer(smooth_idf=False)
exam_bow_fea = vect.fit_transform(exam['SENTENCE']).toarray()
#print len(exam_test)
dictionary = vect.get_feature_names()
dictionary = [ (word) for word in dictionary]
print >> f , (",".join(dictionary))

print >> f,"Loading wrod2vec file"
model = Word2Vec.load('weibodata_vectorB.gem')

#
# print '替换方法1：对于不在字典里的词，开始替换为近义词：'
#
# for sentences in exam_test['SENTENCE']:
#     for word in sentences:
#         print "替换前： ",word
#         i = 0
#         temp = word
#         while ( word not in dictionary and word != " "):  #如果word 不在词典之中
#             try:
#                 word = model.most_similar(positive=[word])[i][0]
#                 print "待选择： " ,word
#                 i += 1                                         #替换word直到在字典
#                 if i == 10:
#                     word = temp
#                     break
#             except:
#                 print word + "not in vocabulary"
#                 break
#         print "替换后： ",word

        #在字典之内的不做替换

print >> f,''.join(u'替换方法2： 直接找出词典中与之最相近的词：')     #另一种替换，等待跑出的结果

list = []

for sentences in exam_test['SENTENCE']:
    temp = ""
    tempWord = ""       #word不可以每次都改变
    sentence = sentences.split(" ")
    print sentence
    for word in sentence:  #对于每一个单词
        if word not in dictionary:  #如果word 不在词典之中
            #print ''.join(u'单词不在词典之中')
            if word == "！":         #空格 则下一个
             #   print ''.join(u'空格跳出循环')
                temp = temp + " " + word
                continue
            if word == "？":
                temp = temp + " " + word
                continue


            origin = 0
            count = 0
            for word_in_dict in dictionary :

                    #print ''.join(u'开始计算不在tfidf字典中的单词与字典中单词的相近程度')
                    #print "count: ",count
                    if count > 20:
                        break

                    #print "尝试计算 " + word + " 与 " + word_in_dict + "的相似度"
                    try:
                        similar = abs(model.similarity(word, word_in_dict))

                    except Exception:
                     #   print word_in_dict + " 或 " + word + " 不在w2v字典，匹配下一个"
                        count += 1
                        continue
                    print >> f,(word + " 与 " + word_in_dict + " 的相似度为：" + str(similar))
                    if similar > origin:
                            origin = similar
                            #print "Before: ",word
                            #print "temp: ",word_in_dict
                            tempWord = word_in_dict         #替换为词典中最相近的词    此时未覆盖原词语，使得最终结果相同
                            #print word + " 被替换为： " + word_in_dict
            word = tempWord
        temp = temp + " " + word

    list.append(temp)

print  >> f,''.join("替换完成，开始计算tfidf:")

exam_test['SENTENCE'].to_csv("origin.csv")
exam_test['SENTENCE'] = list
exam_test['SENTENCE'].to_csv("final.csv")


exam_bow_fea_test = vect.transform(exam_test['SENTENCE']).toarray()
print len(exam_bow_fea_test)

exam_bow_fea_data = exam_bow_fea       #暂未归一化
print len(exam_bow_fea_data)
exam_bow_fea_target = exam['LABEL2']
print len(exam_bow_fea_target)

exam_bow_fea_test_data = exam_bow_fea_test        #暂未归一化
print len(exam_bow_fea_test_data)
exam_bow_fea_test_target = exam_test['LABEL2']
print len(exam_bow_fea_test_target)



print '计算最大熵模型'


print "Training MaxEnt"
clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
clf.fit(exam_bow_fea_data,exam_bow_fea_target)

# print exam_bow_fea_test_target
# print exam_bow_fea_test_data

exam_bow_fea_test_target.to_csv("target_true.csv")
print   >> f,",".join(clf.predict(exam_bow_fea_test_data))
print  >> f,sum(exam_bow_fea_test_target == clf.predict(exam_bow_fea_test_data))/58.0
print  >> f,sum(exam_bow_fea_test_target == clf.predict(exam_bow_fea_test_data))


