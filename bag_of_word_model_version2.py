#encoding=utf8

__author__ = 'jdwang'
__date__ = 'create date: 2016-05-17'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
# 获取该脚本的配置
config = yaml.load(file('/home/jdwang/PycharmProjects/corprocessor/coprocessor/config.yaml'))
config = config['bag_of_word_model_version2']
# 使用什么模型：TFIDF 或者 BOW
CHOICES = config['model']
logging.basicConfig(filename=''.join(config['log_file_path']).lower(), filemode='w', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

logging.debug(config['describe'])
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 设置最大字典大小
MAX_FEATURES = config['max_num_features']

logging.debug('该脚本使用BOW(bag of word） 或者 TFIDF 模型加随机森林做句子分类，分词使用jieba分词工具。')
logging.debug('='*20)

train_data_file_path = config['train_data_file_path']

test_data_file_path = config['test_data_file_path']

logging.debug('加载训练数据文件和测试数据文件...')

train_data = pd.read_csv(train_data_file_path,
                    sep='\t',
                    encoding='utf8',
                    header=0
                    )

test_data = pd.read_csv(test_data_file_path,
                    sep='\t',
                    encoding='utf8',
                    header=0
                    )

logging.debug('the shape of train data:%s'%(str(train_data.shape)))
logging.debug('the shape of test data:%s'%(str(test_data.shape)))

train_data = train_data[train_data['LABEL']!= u'其他#其他']
test_data = test_data[test_data['LABEL']!= u'其他#其他']
logging.debug('-'*20)
logging.debug('去除类别为 其他#其他 的数据，then...')
logging.debug('the shape of train data:%s'%(str(train_data.shape)))
logging.debug('the shape of test data:%s'%(str(test_data.shape)))
# 总共类别数
all_label = train_data['LABEL'].unique()
logging.debug(u'使用类别（%d个）：%s'%(len(all_label),','.join(all_label)))
label2idx = { item:idx for idx,item in enumerate(all_label)}
train_label = train_data['LABEL'].apply(lambda x: label2idx[x])
test_label = test_data['LABEL'].apply(lambda x: label2idx[x])
# print train_data['LABEL']
# print test_data['LABEL']
#
# def preprocessing_data(data):
#     logging.info('Beginning preprocessing the data...')
#     # omit the pos mark and concat the word with ' '
#     # omit the stopwords with pos 'r'(代词)、'y'（代词）
#     # stopwords = set(['r', 'y', 'ude'])
#     # exclude_words = ['嗯', '它', '给力', '呵', '嗨', '呀']
#     # replace the number with mark <NUM>
#     # replace the 拟声词 with mark
#     # data['words'] = data['sentence'].apply(
#     #     lambda line: '|'.join(
#     #             [re.sub(r'.*,m','<NUM>,m',item)
#     #              for item in line.split('|')
#     #              ]
#     #             )
#     # )
#
#     data['words'] = data['words'].apply(
#         lambda line : ' '.join(
#             [item.split(',')[0]
#              for item in line.split('|')
#              if not (item.split(',')[1] in stopwords and item.split(',')[0] not in exclude_words)
#              ]
#         )
#     )
#     logging.debug(data.head())
#     logging.debug(data[-5:])

# 转换输入的格式
train_data['words'] = train_data[config['train_segment_column']].apply(lambda x:' '.join(x.split('|')))
test_data['words'] = test_data[config['test_segment_column']].apply(lambda x:' '.join(x.split('|')))

logging.debug('-'*20)
logging.info('Beginning fit the bag-of-word model')
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
if CHOICES == 'TFIDF':
    logging.debug('使用模型：%s'%(CHOICES))
    vectorizer = TfidfVectorizer(analyzer="word",
                                 token_pattern=u'(?u)\\b\w+\\b',
                                 tokenizer=None,
                                 preprocessor=None,
                                 lowercase=False,
                                 stop_words=None,
                                 max_features=MAX_FEATURES)
else:
    logging.debug('使用模型：%s'%(CHOICES))
    vectorizer = CountVectorizer(analyzer="word",
                                 token_pattern=u'(?u)\\b\w+\\b',
                                 tokenizer=None,
                                 preprocessor=None,
                                 lowercase=False,
                                 stop_words=None,
                                 max_features=MAX_FEATURES)

print train_data.head()
print test_data.head()
train_data_features = train_data['words'].as_matrix()
train_data_features = vectorizer.fit_transform(train_data_features)
vocab = vectorizer.get_feature_names()
print vocab
logging.debug('字典大小:%d'%(len(vocab)))
logging.debug(u'字典详情：%s'%(','.join(vocab)))
train_data_features = train_data_features.toarray()
logging.info('the feature\'shape of train data is: %d,%d' % (train_data_features.shape))


test_data_features = test_data['words'].as_matrix()
test_data_features = vectorizer.transform(test_data_features)
test_data_features = test_data_features.toarray()

logging.info('the feature\'shape of test data is: %d,%d' % (test_data_features.shape))


# print train_data_features
# print test_data_features


# Initialize a Random Forest classifier with 100 trees
n_estimators = 150
logging.debug('Initialize a Random Forest classifier with %d trees'%(n_estimators))
forest = RandomForestClassifier(n_estimators = n_estimators,random_state=0)


forest = forest.fit(train_data_features,train_label)

result = forest.predict(test_data_features)
print result[0]
print all_label[result[0]]

test_data['PREDICT'] = [all_label[item] for item in result]
test_data['IS_CORRECT'] = [ pred==real for pred,real in zip(result,test_label)]
accurary = sum(test_data['IS_CORRECT'])/(len(test_data)*1.0)
logging.debug('准确率为：%f'%(accurary))
print('准确率为：%f'%(accurary))

probs = forest.predict_proba(test_data_features)
# print np.argsort(probs)
# print [prob[index] for index,prob in zip(np.argsort(probs)[:,-2],probs)]
# quit()
test_data['best1/best2'] = np.array([prob[index] for index,prob in zip(np.argsort(probs)[:,-1],probs)])/np.array([prob[index] for index,prob in zip(np.argsort(probs)[:,-2],probs)])

counter = 1
while counter <= len(all_label):
    test_data['best_%d' % (counter)] =[ '%f,%s'%(prob[index],all_label[index]) for index,prob in zip(np.argsort(probs)[:,-counter],probs)]

    counter += 1
# quit()
# # print [prob[index] for index,prob in zip(np.argsort(probs)[:,-1],probs)]
# # print [prob[index] for index,prob in zip(np.argsort(probs)[:,-2],probs)]
# # print np.array([prob[index] for index,prob in zip(np.argsort(probs)[:,-1],probs)])/np.array([prob[index] for index,prob in zip(np.argsort(probs)[:,-2],probs)])
# print test.head()

result_file_path = ''.join(config['result_file_path'])
# output_feature = ['sentence','words','label_name','result','is_correct','best1/best2']


# test.filter(regex='|'.join(output_feature)+'|best_.*').to_csv(result_file_path,
#                                                               sep='\t',
#                                                               encoding = 'utf8',
#                                                               index = None
#                                                               )
test_data.to_csv(result_file_path,
                 sep='\t',
                 encoding='utf8',
                 index=None
                 )




