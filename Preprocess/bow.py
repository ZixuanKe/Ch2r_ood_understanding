#coding: utf-8
import  pandas as pd
import jieba


'''
    简单的分词模块

'''

train_data = pd.read_csv(
    'v2.3_train_S_1518.csv',
    sep='\t',
    encoding='utf8',
    header=0
)

test_data = pd.read_csv(
    'v2.3_test_S_131.csv',
    sep='\t',
    encoding='utf8',
    header=0
)

train_data['WORDS'] = [" ".join(jieba.cut(sentence)) for sentence in train_data['SENTENCE']]
test_data['WORDS'] = [" ".join(jieba.cut(sentence)) + " " for sentence in test_data['SENTENCE']]

train_data.to_csv(
    "train_seg.csv",
    sep='\t',
    encoding='utf8',

)

test_data.to_csv(
    "test_seg.csv",
    sep='\t',
    encoding='utf8',

)