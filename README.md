# Ch2r_ood_understanding

对CH2r进行两只两种方法分类


1. 二阶段法：(teo-phrase)


    1) 直接进行16个类分类


    2) 先对4个小类分类 再进行小类细分

2. 最大熵法(ME(TFIDF+OOV))


    1) 特征提取： TFIDF


    2) 字典外的词(OOV): 替换与字典最近的词 基于word2vec


    3) 模型： 最大熵模型

4. CNN(cnn.py)


    1) seg-CNN


    2) bow-CNN


5. RF(random_forest.py)

6. preprocess 预处理模块 分别将语料库预处理为


    BOC(Characteristic) --- boc.py


    BOW(Word)  --- bow.py

参考论文：
    (Johnson and Zhang, NAACL 2015) Effective Use of Word Order for Text Categorization with Convolutional Neural Networks

语料库：


    含有其原文以及交叉验证（按类别取等量）后的交叉验证数据集（BOW编码）


    1. AIML
    2. CCL
