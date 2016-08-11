# Ch2r_ood_understanding

对CH2r进行两只两种方法分类


1. 二阶段法：(teo-phrase)
    1) 直接进行16个类分类
    2) 先对4个小类分类 再进行小类细分

2. 最大熵法(ME(TFIDF+OOV))
    1) 特征提取： TFIDF
    2) 字典外的词(OOV): 替换与字典最近的词 基于word2vec
    3) 模型： 最大熵模型
