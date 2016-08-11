#coding: utf-8
import  pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import yaml
from keras.utils import np_utils, generic_utils
import pickle
from keras.models import Sequential, Model
from keras.layers import Embedding, Convolution2D, Input, Activation, MaxPooling2D, Reshape, Dropout, Dense, \
    Flatten, Merge
from keras.optimizers import SGD
from keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder
import numpy as np;
np.random.seed(1337)  # for reproducibility
import random

#
# config = yaml.load(file('config_my_cnn.yaml'))	#读取yaml配置文件
# config = config['OriginBow']						#以字典的方式读取2


'''
    cnn 结构可以随时更改
    目前是 一个随机选取方案 的结果


'''



nb_pool = [2,1]
nb_classes = 24



def onehotcoder(train_data,test_data):

    '''
    对应论文中的 seg编码
    :param train_data: 训练数据
    :param test_data:   测试数据
    :return:
    '''

    vect = CountVectorizer()
    train_data_bow_fea = vect.fit_transform(train_data['WORDS']).toarray()
    # 规定4维输入，必须先转化[长度，1，宽度，1]
    test_data_bow_fea = vect.transform(test_data['WORDS']).toarray()

    length = len(vect.vocabulary_)
    values = []
    for i in range(10):
        values.append(length)
    print len(values)
    code = OneHotEncoder(categorical_features=np.array([1,2,3,4,5,6,7,8,9,10]),n_values=values)   #10个类别 每个类别有字典总数种可能
    train_feature = code.fit_transform(train_data_bow_fea).toarray()   #编码
    test_feature = code.transform(test_data_bow_fea).toarray()

    # print "训练集："
    # print "每个词的维度：",code.n_values_

    train_onehot = []
    # print "单词总数：",len(train_feature)
    # print "每行总长度", len(train_feature[1]) * 935 - 1

    for i in range(len(train_feature)):
        train_one_hot_col = []
        t = 0
        while True:
            # print "剩下 " + str( (len( train_feature[i])) - t ) +  " 维"
            if ((len( train_feature[i])) - t)  < 0:
                break
            a = train_feature[i][t:t+935]
            t += 935
            b = train_feature[i][t:t+935]
            c = [a[m]+b[m] for m in range(min(len(a),len(b)))]    #2区域内相加
            for k in c:
                train_one_hot_col.append(k)
        train_onehot.append(train_one_hot_col)

    print "最终维度：",len(train_onehot[0])


    # print "测试集："
    # print "每个词的维度：",code.n_values_

    test_onehot = []
    # print "单词总数：",len(test_feature)
    # print "每行总长度", len(test_feature[1]) * 935 - 1

    for i in range(len(test_feature)):
        test_one_hot_col = []
        t = 0
        while True:
            # print "剩下 " + str( (len( test_feature[i])) - t ) +  " 维"
            if ((len( test_feature[i])) - t)  < 0:
                break
            a = test_feature[i][t:t+935]
            t += 935
            b = test_feature[i][t:t+935]
            c = [a[m]+b[m] for m in range(min(len(a),len(b)))]    #2区域内相加
            for k in c:
                test_one_hot_col.append(k)
        test_onehot.append(test_one_hot_col)

    print "最终维度：",len(test_onehot[0])
    print len(test_onehot)

    return train_onehot,test_onehot


def build(layer1,layer2,hidden1,hidden2,length,width,lr=0.001 ,decay=1e-6,momentum=0.9):
    '''
    开始构建CNN网络
    :param layer1: 第一层网络 卷积核数量
    :param layer2: 第二层网络 卷积核数量
    :param hidden1:  第一个隐藏层网络 卷积核数量
    :param hidden2: 第二个隐藏层网络 卷积核数量
    :param length:  输入长度
    :param width:   输入宽度
    :param lr:  学习率
    :param decay:   学习率衰减
    :param momentum:
    :return:    搭建好的CNN模型
    '''
    #16*5*1

    layer1_model1=Sequential()
    layer1_model1.add(Convolution2D(layer1, 2, 1,
                            border_mode='valid',
                            input_shape=(1, length, 1)))
    layer1_model1.add(Activation('tanh'))
    layer1_model1.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))

    #16*10*1
    layer1_model2=Sequential()
    layer1_model2.add(Convolution2D(layer1, 4, 1,
                            border_mode='valid',
                            input_shape=(1, length, 1)))
    layer1_model2.add(Activation('tanh'))
    layer1_model2.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))

    #16*20*1
    layer1_model3=Sequential()
    layer1_model3.add(Convolution2D(layer1, 6, 1,
                            border_mode='valid',
                            input_shape=(1, length, 1)))
    layer1_model3.add(Activation('tanh'))
    layer1_model3.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))



    model = Sequential()

    model.add(Merge([layer1_model2,layer1_model1,layer1_model3], mode='concat',concat_axis=2))#merge



    model.add(Convolution2D(layer2,3,1))#layer2  32*5*1
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))
    model.add(Dropout(0.25))

    model.add(Flatten()) #平铺

    model.add(Dense(hidden1)) #Full connection 1:  1000
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))


    model.add(Dense(hidden2)) #Full connection 2:   200
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])

    #初始化应该在return 之前

    return model
#

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

    # 测试集"""
    train_data_bow_fea,train_data_label = load_data("v2.3_train_Sa_word_seg_i1_dev_830.csv")
    test_data_bow_fea,test_data_label = load_data("v2.3_train_Sa_word_seg_i1_val_76.csv")

    train_data_bow_fea_v1,train_data_label_v1 = load_data("v2.3_train_Sa_word_seg_i2_dev_555.csv")
    test_data_bow_fea_v1,test_data_label_v1 = load_data("v2.3_train_Sa_word_seg_i2_val_275.csv")
    # #


    train_data_bow_fea_v2,train_data_label_v2 = load_data("v2.3_train_Sa_word_seg_i3_dev_553.csv")
    test_data_bow_fea_v2,test_data_label_v2 = load_data("v2.3_train_Sa_word_seg_i3_val_277.csv")


    train_data_bow_fea_v3,train_data_label_v3 = load_data("v2.3_train_Sa_word_seg_i4_dev_552.csv")
    test_data_bow_fea_v3,test_data_label_v3 = load_data("v2.3_train_Sa_word_seg_i4_val_278.csv")


    sentence_width = len(train_data_bow_fea)
    sentence_length = len(train_data_bow_fea[1])

    sentence_width_v1 = len(train_data_bow_fea_v1)
    sentence_length_v1 = len(train_data_bow_fea_v1[1])

    sentence_width_v2 = len(train_data_bow_fea_v2)
    sentence_length_v2 = len(train_data_bow_fea_v2[1])

    sentence_width_v3 = len(train_data_bow_fea_v3)
    sentence_length_v3 = len(train_data_bow_fea_v3[1])

    print sentence_width
    print sentence_length

    train_data_bow_fea = np.array(train_data_bow_fea).reshape(len(train_data_bow_fea), 1, len(train_data_bow_fea[1]), 1)
    # 规定4维输入，必须先转化[长度，1，宽度，1]
    test_data_bow_fea = np.array(test_data_bow_fea).reshape(len(test_data_bow_fea), 1, len(test_data_bow_fea[1]), 1)


    train_data_bow_fea_v1 = np.array(train_data_bow_fea_v1).reshape(len(train_data_bow_fea_v1), 1, len(train_data_bow_fea_v1[1]), 1)
    # 规定4维输入，必须先转化[长度，1，宽度，1]
    test_data_bow_fea_v1 = np.array(test_data_bow_fea_v1).reshape(len(test_data_bow_fea_v1), 1, len(test_data_bow_fea_v1[1]), 1)


    train_data_bow_fea_v2 = np.array(train_data_bow_fea_v2).reshape(len(train_data_bow_fea_v2), 1, len(train_data_bow_fea_v2[1]), 1)
    # 规定4维输入，必须先转化[长度，1，宽度，1]
    test_data_bow_fea_v2 = np.array(test_data_bow_fea_v2).reshape(len(test_data_bow_fea_v2), 1, len(test_data_bow_fea_v2[1]), 1)

    train_data_bow_fea_v3 = np.array(train_data_bow_fea_v3).reshape(len(train_data_bow_fea_v3), 1, len(train_data_bow_fea_v3[1]), 1)
    # 规定4维输入，必须先转化[长度，1，宽度，1]
    test_data_bow_fea_v3 = np.array(test_data_bow_fea_v3).reshape(len(test_data_bow_fea_v3), 1, len(test_data_bow_fea_v3[1]), 1)
    #改造： 维度也卷积
    #改造： 参数改变等

    print  '句子数：',sentence_width
    print  '维度总数：',sentence_length

    label_train = train_data_label
    label_train = np_utils.to_categorical(label_train, 24)  # 必须使用固定格式表示标签
    label_test = test_data_label
    label_test = np_utils.to_categorical(label_test, 24)  # 必须使用固定格式表示标签
	
	

    label_train_v1 = train_data_label_v1
    label_train_v1 = np_utils.to_categorical(label_train_v1, 24)  # 必须使用固定格式表示标签
    label_test_v1 = test_data_label_v1
    label_test_v1 = np_utils.to_categorical(label_test_v1, 24)  # 必须使用固定格式表示标签
	
    label_train_v2 = train_data_label_v2
    label_train_v2 = np_utils.to_categorical(label_train_v2, 24)  # 必须使用固定格式表示标签
    label_test_v2 = test_data_label_v2
    label_test_v2 = np_utils.to_categorical(label_test_v2, 24)  # 必须使用固定格式表示标签

    label_train_v3 = train_data_label_v3
    label_train_v3 = np_utils.to_categorical(label_train_v3, 24)  # 必须使用固定格式表示标签
    label_test_v3 = test_data_label_v3
    label_test_v3 = np_utils.to_categorical(label_test_v3, 24)  # 必须使用固定格式表示标签


    # layer1_model1 = [10,9,11]
    # layer2_model = [30,31,29]
    # hidden1_model = [1000,980,1020]
    # hidden2_model = [100,80,120]
    #
    # c = 5

    # layer1_model1 = [5, 6, 4]
    # layer2_model = [30, 31, 29]
    # hidden1_model = [1000, 980, 1020]
    # hidden2_model = [450, 430, 470]
    #
    # c = 4

    # layer1_model1 = [10, 11, 9]
    # layer2_model = [30, 31, 29]
    # hidden1_model = [1000, 980, 1020]
    # hidden2_model = [450, 430, 470]
    #
    # c = 3

    # layer1_model1 = [10, 11, 9]
    # layer2_model = [15, 14, 16]
    # hidden1_model = [1000, 980, 1020]
    # hidden2_model = [300, 280, 320]
    #
    # c = 2

    layer1_model1 = [10, 11, 9]
    layer2_model = [30,31, 29]
    hidden1_model = [1000, 980, 1020]
    hidden2_model = [300, 280, 320]

    c = 1
    print c
    plan = []
    for i in range(0, len( layer1_model1)):
        for j in range(0, len( layer2_model)):
            for k in range(0, len( layer2_model)):
                for m in range(0, len( layer2_model)):
                   plan.append([layer1_model1[i],layer2_model[j],hidden1_model[k],hidden2_model[m]])

    random.shuffle(plan)





    u = 0

    # for layer1 in layer1_model1:    #4,6
    #     for layer2 in layer2_model: #[6,8]
    #         for hidden1 in hidden1_model:
    #             for hidden2 in hidden2_model:
    for i in range(20):

                        layer1 = plan[i][0]
                        layer2 = plan[i][1]
                        hidden1 = plan[i][2]
                        hidden2 = plan[i][3]

                        f = open('result.txt','a')

                        print 'layer1: ', layer1
                        print 'layer2: ', layer2
                        print 'hidden1: ', hidden1
                        print 'hidden2: ', hidden2

                        print >> f, 'layer1: ', layer1
                        print  >> f,'layer2: ', layer2
                        print  >> f,'hidden1: ', hidden1
                        print  >> f,'hidden2: ', hidden2



                        #不同卷积核意味着不同权值

                        model = build( layer1,layer2,hidden1,hidden2,sentence_length,sentence_width)


                        model.fit([train_data_bow_fea,train_data_bow_fea,train_data_bow_fea],label_train, batch_size=32, nb_epoch=30,shuffle=True,verbose=1,validation_split=0)


                        print '测试准确率：'
                        print model.metrics_names
                        print model.evaluate([test_data_bow_fea,test_data_bow_fea,test_data_bow_fea],label_test,show_accuracy=True)

                        print   >> f,'测试准确率：'
                        print   >> f,model.metrics_names
                        print   >> f,model.evaluate([test_data_bow_fea, test_data_bow_fea, test_data_bow_fea], label_test, show_accuracy=True)

                        acc = model.evaluate([test_data_bow_fea,test_data_bow_fea,test_data_bow_fea],label_test,show_accuracy=True)[1]

						
						
                        #v1
                        model = build( layer1,layer2,hidden1,hidden2,sentence_length_v1,sentence_width_v1)
                        model.fit([train_data_bow_fea_v1,train_data_bow_fea_v1,train_data_bow_fea_v1],label_train_v1, batch_size=32, nb_epoch=30,shuffle=True,verbose=1,validation_split=0)
                        acc_v1 = model.evaluate([test_data_bow_fea_v1,test_data_bow_fea_v1,test_data_bow_fea_v1],label_test_v1,show_accuracy=True)[1]
                        #v2
                        model = build( layer1,layer2,hidden1,hidden2,sentence_length_v2,sentence_length_v2)
                        model.fit([train_data_bow_fea_v2,train_data_bow_fea_v2,train_data_bow_fea_v2],label_train_v2, batch_size=32, nb_epoch=30,shuffle=True,verbose=1,validation_split=0)
                        acc_v2 = model.evaluate([test_data_bow_fea_v2,test_data_bow_fea_v2,test_data_bow_fea_v2],label_test_v2,show_accuracy=True)[1]

                        #v3
                        model = build( layer1,layer2,hidden1,hidden2,sentence_length_v3,sentence_length_v3)
                        model.fit([train_data_bow_fea_v3,train_data_bow_fea_v3,train_data_bow_fea_v3],label_train_v3, batch_size=32, nb_epoch=30,shuffle=True,verbose=1,validation_split=0)
                        acc_v3 = model.evaluate([test_data_bow_fea_v3,test_data_bow_fea_v3,test_data_bow_fea_v3],label_test_v3,show_accuracy=True)[1]
										
						
                        import csv

                        csvfile = file('result_word&charact_Best' + str(c) + '_Random.csv', 'a')
                        writer = csv.writer(csvfile)
                        if u == 0:
                            writer.writerow(['layer1', 'layer2', 'hidden1','hidden2','val_acc','test_acc'])
                            u += 1

                        data = [
                            (layer1, layer2, hidden1,hidden2,acc,((acc_v1 + acc_v2 + acc_v3)/(3*1.0)) )
                        ]
                        writer.writerows(data)
                        csvfile.close()

                        # import csv
                        #
                        # csvfile = file('result_word.csv', 'a')
                        # writer = csv.writer(csvfile)
                        # if u == 0:
                        #     writer.writerow(['layer1', 'layer2', 'hidden1','hidden2','val_acc','test_acc'])
                        #     u += 1
                        #
                        # data = [
                        #     (layer1, layer2, hidden1,hidden2,"",model.evaluate([test_data_bow_fea,test_data_bow_fea,test_data_bow_fea],label_test,show_accuracy=True)[1]),
                        # ]
                        # writer.writerows(data)
                        # csvfile.close()

                        f.close()