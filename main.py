#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

__author__ = 'jdwang'
__date__ = 'create date: 2016-05-29'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from dateutil.parser import parse
import pandas as pd
import jieba

#
# print 'Loading Data'
# exam = pd.read_table('new_train_all.csv',
#                      converters={'date': parse},encoding = 'utf-8')
#
#
# exam_test = pd.read_table('new_ood_labeled.csv',
#                      converters={'date': parse},encoding = 'utf-8')
#
# print len(exam)
# print len(exam_test)

#2、分词

#数字变number
#的字被去掉
#部分词没有在字典出现 、
#替换的大部分是停用词

# exam = exam.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD'],axis=1)
# # exam_test = exam_test.drop(['SEGMENT','SEGMENT_FULL','SEGMENT_EVERYWORD','SEGMENT_OOV','SEGMENT_OOV_EVERYWORD'],axis=1)
# exam['SENTENCE'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in exam['SENTENCE']]
# exam_test['SENTENCE'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in exam_test['SENTENCE']]
#
#


config = yaml.load(file('./config.yaml'))	#读取yaml配置文件
config = config['main']						#以字典的方式读取2
logging.basicConfig(filename=''.join(config['log_file_path']), filemode='w',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()

#可保存为日志文件进行管理

print('=' * 30)
# print config['describe']
print('=' * 30)
print 'start running!'
logging.debug('=' * 30)		
logging.debug(config['describe'])
logging.debug('=' * 30)
logging.debug('start running!')
logging.debug('=' * 20)


import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv(
    config['train_data_file_path'],
    sep='\t',
    encoding='utf8',
    header=0
)

test_data = pd.read_csv(
    config['test_data_file_path'],
    sep='\t',
    encoding='utf8',
    header=0
)

logging.debug('train data shape is :%s'%(str(train_data.shape)))
print('train data shape is :%s'%(str(train_data.shape)))

logging.debug('test data shape is :%s'%(str(test_data.shape)))
print('test data shape is :%s'%(str(train_data.shape)))
logging.debug('-' * 20)
# 去除类别 其他#其他
logging.debug('去除类别 其他#其他')
train_data = train_data[train_data['LABEL']!=u'其他#其他']
test_data = test_data[test_data['LABEL']!=u'其他#其他']
logging.debug('train data shape is :%s'%(str(train_data.shape)))
print('train data shape is :%s'%(str(train_data.shape)))

logging.debug('test data shape is :%s'%(str(test_data.shape)))
print('test data shape is :%s'%(str(train_data.shape)))
logging.debug('-' * 20)

train_data = train_data[['LABEL','SENTENCE']]
test_data = test_data[['LABEL','SENTENCE']]

index_to_label = list(train_data['LABEL'].unique())
logging.debug(u'总共类别数:%d,分别为:%s'%(len(index_to_label),','.join(index_to_label)))
print('总共类别数:%d'%(len(index_to_label)))

label_to_index = {label:idx for idx,label in enumerate(index_to_label)}

train_data['LABEL_INDEX'] = train_data['LABEL'].map(label_to_index)
test_data['LABEL_INDEX'] = test_data['LABEL'].map(label_to_index)
# print train_data.head()


logging.debug('=' * 20)
logging.debug('对数据进行分词...')
logging.debug('-' * 20)

sentence_to_seg = lambda x: jieba.cut(x,cut_all=True)

train_data['WORDS'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in  train_data['SENTENCE']]
test_data['WORDS'] = [' '.join(jieba.cut(sentence,cut_all=True)) for sentence in  test_data['SENTENCE']]

# train_data['WORDS'] = train_data['SENTENCE'].apply(sentence_to_seg)
# test_data['WORDS'] = test_data['SENTENCE'].apply(sentence_to_seg)
print train_data.head()

logging.debug('=' * 20)
logging.debug('开始生成特征向量...')

vectorizer = CountVectorizer(analyzer="word",
                             token_pattern=u'(?u)\\b\w+\\b',
                             tokenizer=None,
                             preprocessor=None,
                             lowercase=False,
                             stop_words=None,
                             max_features=config['max_features'])

print test_data.head()
train_X_features = vectorizer.fit_transform(train_data['WORDS'].as_matrix()).toarray(

)


vocabulary = vectorizer.get_feature_names()
logging.debug(u'字典大小:%d个词,有:%s'%(len(vocabulary),','.join(vocabulary)))
# print(u'字典大小:%d,有:%s'%(len(vocabulary),','.join(vocabulary)))

logging.debug('train X shape is :%s'%(str(train_X_features.shape)))
print('train X shape is :%s'%(str(train_X_features.shape)))

logging.debug('=' * 20)
logging.debug(u'计算概率')
logging.debug('注意:如果一个词在一个句子中出现多次,也只算一次,即这里计算的是,这个词在多少个句子中出现的次数')

row,col = train_X_features.shape
# 若一个词在句子中出现多次,只算一次
train_X_features = np.asarray([item>0 for item in train_X_features.flatten()],dtype=int).reshape(row,col)

words_total_count = sum(train_X_features.flatten())
logging.debug('训练库中,词的总计数为:%d'%(words_total_count))
print('训练库中,词的总计数为:%d'%(words_total_count))

logging.debug('-' * 20)
# 统计每个词的出现次数,如果一个词在一个句子中出现多次,也只算一次,即这里计算的是,这个词在多少个句子中出现的次数
logging.debug('统计每个词的出现次数,如果一个词在一个句子中出现多次,也只算一次,即这里计算的是,这个词在多少个句子中出现的次数')
get_word_count = lambda x: sum(x)
word_counts =  np.sum(train_X_features,axis=0)

p_word = word_counts/(1.0*words_total_count)
logging.debug(u'最大词频为:%f,次数为:%d,该词为:%s'%(max(p_word),max(word_counts),vocabulary[np.argmax(word_counts)]))
# print(u'最大词频为:%f,次数为:%d,该词为:%s'%(max(p_word),max(word_counts),vocabulary[np.argmax(word_counts)]))

logging.debug('-' * 20)
logging.debug('计算词和各个类的共现次数,以及每个类的句子数...')

print('计算词和各个类的共现次数...')
# count(word,class)
count_word_class = []
# count(class)
count_class = []
for label in index_to_label:
    logging.debug('-' * 10)
    logging.debug(u'处理类别:%s'%(label))
    # print(u'处理类别:%s'%(label))
    # 计算相应类别的句子
    index = (train_data['LABEL'] == label).as_matrix()
    sentences = train_X_features[index]
    print len(sentences)
    logging.debug('句子数为:%d'%(len(sentences)))
    print('句子数为:%d'%(len(sentences)))
    count_class.append(len(sentences))
    count_word_class.append(np.sum(sentences,axis=0))

# count(class)
count_class = np.asarray(count_class)
# P(class)
p_class = count_class/(1.0*len(train_data))
# P(class|word)
p_class_on_word = count_word_class/(word_counts*1.0)
p_class_on_word = p_class_on_word.transpose()

logging.debug('-' * 20)
logging.debug('计算 P(class|word)/P(class)')

print p_class_on_word[0]
print p_class
# P(class|word)/P(class)
p_rate = p_class_on_word/p_class
print p_rate[0]
logging.debug('计算 log( P(class|word)/P(class) )')
# log( P(class|word)/P(class) )
log_p_rate = np.log(p_rate)
print log_p_rate[0]

# P(class|word) * log( P(class|word)/P(class) )
p_ent = log_p_rate * p_class_on_word
p_ent = np.nan_to_num(p_ent)
print p_ent[0]
# 期望交叉熵
entroy = np.sum(p_ent,axis=1)
print entroy[0]

print p_word[0]
# 结果 = 期望交叉熵 * P(word)

# 论文直接使用词频*熵,则将会导致词频大的词权重很大,
# 即:entroy = p_word * entroy
# 改进:使用sigmoid函数进行平滑
# 或者不使用词频,效果也更好
def sigmoid(x):
    return 1/(1+np.exp(-x))
# entroy = sigmoid(p_word) * entroy
print entroy[0]

logging.debug('=' * 20)

logging.debug('进行特征词选择..')
logging.debug('-' * 20)
sort_index = np.argsort(entroy)[-1::-1]
vocabulary = np.asarray(vocabulary)
# print ','.join(vocabulary[sort_index])
# print entroy[sort_index]
logging.debug(u'期望交叉熵top 10:%s'%(','.join(vocabulary[sort_index[:10]])))
logging.debug('大小分别为:%s'%(entroy[sort_index[:10]]))

logging.debug('-' * 20)
keywords = vocabulary[sort_index[:config['max_keywords']]]

logging.debug('选取%d个词作为关键词,实际为:%d个'%(config['max_keywords'],len(keywords)))
# print('选取%d个词作为关键词,实际为:%d'%(config['max_keywords'],len(keywords)))
logging.debug(u'关键词分别为(按权重大到小):%s'%(','.join(keywords)))
# print(u'关键词分别为(按权重大到小):%s'%(','.join(keywords)))
logging.debug('-' * 20)




logging.debug('=' * 20)
logging.debug('生成TFIDF特征向量...')
# TFIDF 字典
tfidf_vocabulary = {item:idx for idx,item in enumerate(keywords)}

tfidf_vectorizer = TfidfVectorizer(analyzer="word",
                                token_pattern=u'(?u)\\b\w+\\b',
                                 tokenizer=None,
                                 preprocessor=None,
                                 lowercase=False,
                                 stop_words=None,
                                 vocabulary = tfidf_vocabulary,
                             max_features=config['max_keywords'])

exam_bow_fea = tfidf_vectorizer.fit_transform(train_data['WORDS'].as_matrix()).toarray()
print "test: ",len(exam_bow_fea)

f = open("result.txt","w")
dictionary = tfidf_vectorizer.get_feature_names()
dictionary = [ (word) for word in dictionary]

print "dictionary length: ",len(dictionary)
print len(test_data['LABEL'])


print >> f , (",".join(dictionary))

print >> f,"Loading wrod2vec file"
model = Word2Vec.load('weibodata_vectorB.gem')

print >> f,''.join(u'替换方法2： 直接找出词典中与之最相近的词：')     #另一种替换，等待跑出的结果

list = []

for sentences in test_data['WORDS']:
    temp = ""
    tempWord = ""       #word不可以每次都改变
    sentence = sentences.split(" ")
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

test_data['WORDS'].to_csv("origin.csv")
test_data['WORDS'] = list
test_data['WORDS'].to_csv("final.csv")


exam_bow_fea_test = tfidf_vectorizer.transform(test_data['WORDS'].as_matrix()).toarray()

print len(exam_bow_fea_test)

exam_bow_fea_target = train_data['LABEL']
print len(exam_bow_fea_target)

exam_bow_fea_test_target = test_data['LABEL']
print len(exam_bow_fea_test_target)



print '计算最大熵模型'


print "Training MaxEnt"
# rf = RandomForestClassifier(n_estimators=200)
# clf = rf
clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
clf.fit(exam_bow_fea,exam_bow_fea_target)

# print exam_bow_fea_test_target
# print exam_bow_fea_test_data
exam_bow_fea_test_target = test_data['LABEL']
print len(exam_bow_fea_test_target)
print len(exam_bow_fea_test)
print len(test_data)

exam_bow_fea_test_target.to_csv("target_true.csv")
print   >> f,",".join(clf.predict(exam_bow_fea_test))
print  >> f,sum(exam_bow_fea_test_target == clf.predict(exam_bow_fea_test))
print  >> f,sum(exam_bow_fea_test_target == clf.predict(exam_bow_fea_test))/(len(test_data)*1.0)

