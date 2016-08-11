#coding: utf-8
import  pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import yaml
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier



'''
    尝试使用randomforest进行分类 与 CNN分类进行比对

'''
def load_data(file_name):
    import csv

    csvfile = file(file_name, 'rb')
    reader = csv.reader(csvfile)

    label = []
    data = []
    for line in reader:
        label.append(line[0])
        data.append(line[1:len(line)])

    # print label
    # print data
    csvfile.close()
    return data,label



if __name__ == '__main__':

    data = pd.read_csv(
        "v2.3_test_Sa_79.csv",
        sep='\t',
        encoding='utf8',
        header=0
    )
    f = open("result.txt",'a')
    train_data_bow_fea_bow,train_data_label_bow = load_data("v2.3_train_Sa_word_seg_i1_dev_830.csv")
    test_data_bow_fea_bow,test_data_label_bow = load_data("v2.3_train_Sa_word_seg_i1_val_76.csv")
    print "拼接 i1，卷积层"
    with open("TrainSet_2+281_feature_d1.pickle","rb") as file:
        train_data_bow_fea = pickle.load(file)
        train_data_label = pickle.load(file)
        test_data_bow_fea = pickle.load(file)
        test_data_label = pickle.load(file)
    #
    # train_data_bow_fea_bow = preprocessing.minmax_scale(train_data_bow_fea_bow)
    # test_data_bow_fea_bow = preprocessing.minmax_scale(test_data_bow_fea_bow)

    #拼接用
    print "length1: " + str(len(train_data_bow_fea_bow[0]))
    print "length2: " + str(len(train_data_bow_fea[0]))
    print len(train_data_bow_fea_bow)
    print len(train_data_bow_fea)

    train_length = len(train_data_bow_fea_bow[0]) + len(train_data_bow_fea[0])
    test_length = len(test_data_bow_fea_bow[0]) + len(test_data_bow_fea[0])

    train_weigth = len(train_data_bow_fea_bow)
    test_weigth = len(test_data_bow_fea_bow)

    train_data_bow_fea = np.concatenate((train_data_bow_fea,train_data_bow_fea_bow),axis=1)
    test_data_bow_fea = np.concatenate((test_data_bow_fea,test_data_bow_fea_bow),axis=1)

    train_data_bow_fea.reshape(train_length,train_weigth)
    test_data_bow_fea.reshape(test_length,test_weigth)

    print  "length合并: " + str(len(train_data_bow_fea[0]))

    train  = train_data_bow_fea
    test  = test_data_bow_fea

    index_to_label = [
            u'其它#骂人',
            u'导购#不成交',
			u'导购#不理解',
			u'导购#开始',
            u'导购#成交',
			u'导购#更换',
			u'导购#结束',
			u'导购#详情',
            u'表态#不满',
			u'表态#否定',
			u'表态#满意',
            u'表态#犹豫',
			u'表态#疑问',
			u'表态#肯定',
			u'表态#附和',
			u'表态#随便',
            u'社交义务#不用谢',
			u'社交义务#接受道歉',
			u'社交义务#致谢',
            u'社交义务#道歉',
			u'社交义务#问候',
            u'闲聊#天气',
			u'闲聊#时间',
			u'闲聊#身份信息'
        ]

    for n in [1000]:
        clf  = RandomForestClassifier(n_estimators=n) #随机森林
        # clf.fit(train_data_bow_fea,train_data['LABEL'])

        clf.fit(train,train_data_label)
        print  >> f ,sum(test_data_label ==  clf.predict(test)) / (1.0 * len(test_data_label))
        print  sum(test_data_label ==  clf.predict(test)) / (1.0 * len(test_data_label))


        #bad case 输出
        predict =   clf.predict(test_data_bow_fea)
        for i in range(len(test_data_label)):
            if test_data_label[i] != predict[i]:
                print data['SENTENCE'][i] + "\t" +  index_to_label[int(predict[i])] + "\t" + index_to_label[int(test_data_label[i])]