#Ch2r_ood_understanding

---
本文档为论文[限定领域口语对话系统中超出领域话语的对话行为识别](https://zixuanke.github.io/docs/%E9%99%90%E5%AE%9A%E9%A2%86%E5%9F%9F%E5%8F%A3%E8%AF%AD%E5%AF%B9%E8%AF%9D%E7%B3%BB%E7%BB%9F%E4%B8%AD%E8%B6%85%E5%87%BA%E9%A2%86%E5%9F%9F%E8%AF%9D%E8%AF%AD%E7%9A%84%E5%AF%B9%E8%AF%9D%E8%A1%8C%E4%B8%BA%E8%AF%86%E5%88%AB.pdf)的部分实验代码。代码基于Python，需要用到的外部库有：

> * Keras（搭建神经网络）
> * Scikit-learn（最大熵，随机森林）
> * gensim(使用word2vec替换字典外的词)

实验涉及的方法主要有
> * 二阶段法（two-phase）
> * 最大熵法（ME(TFIDF+OOV)）
> * 随机森林（RF(random_forest.py)）
> * CNN(cnn.py)

语料库简介  
   [语料库](https://github.com/ZixuanKe/Ch2r_ood_understanding/tree/master/corpus)中有两个语料库可供选择：
   > * AIML语料库（人造数据集）
  > * CCL语料库（实际测试用到的数据集）

标签格式为：

> categoryA # categoryB

即 **大类维度为A，小类维度为B**


其中 **大类共4类，小类共16类**

实验方法  
预处理模块  
   [预处理](https://github.com/ZixuanKe/Ch2r_ood_understanding/blob/master/Preprocess)中有两个预处理脚本可供选择：
   > * BOC（Bag-of-character 即按字划分，制造“字袋”）
  > * BOW（Bag-of-word 即按词划分，制造“词袋”）

二阶段法  
我们将分类切割成两部分，首先进行4个大类的分类，在大类的基础上，再对大类下的小类进行细分 
> 这样做的合理性，在部分比赛参赛选手的做法中得到证实。理由是我们认为大类分类比小类分类更加容易，在大类之内进行小类分类，可以使得小类分类时范围减少，减少小类分类的难度。然而这样也有不合理性，比如，大类分类出错，则小类分类则无机会再分对，也即误差的传递性。

> 参考论文： [Splusplus: A Feature-Rich Two-stage Classifier for Sentiment Analysis of Tweets](http://www.aclweb.org/anthology/S/S15/S15-2.pdf#page=557)

在代码中，针对每个大类对应的小类，重新训练了各自的分类器：
```python
resultData,resultTarget = findAllTrainning('attitude',exam_bow_fea_data)         #找到其大类的所有小类
gb1 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('shopping',exam_bow_fea_data)         #找到其大类的所有小类
gb2 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('chatting',exam_bow_fea_data)         #找到其大类的所有小类
gb3 = sub_classfier(resultData,resultTarget)
resultData,resultTarget = findAllTrainning('trouble',exam_bow_fea_data)         #找到其大类的所有小类
gb4 = sub_classfier(resultData,resultTarget)
```
最大熵法  
使用最大熵模型直接分类作为对照组
>* 最大熵模型在许多文本分类问题中都表现了他优越的性能，这里我们利用他作为对照组，观察后面CNN和RF的效果

> 参考论文： [使用最大熵模型进行中文文本分类](http://www.cnki.net/KCMS/detail/detail.aspx?QueryID=4&CurRec=1&recid=&filename=JFYZ200501013&dbname=CJFD2005&dbcode=CJFQ&pr=&urlid=&yx=&v=MjkxMDVMRzRIdFRNcm85RVo0UjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSTHlmYitSckZ5L2hVYnpPTHl2U2Q=)

>* 当逻辑回归用于多分类问题时，可将损失函数改为交叉熵之后，则其成为最大熵模型[LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)


>* 为了提高分类精度，针对部分在字典外的词，使用word2vec用外部语料（论文中使用SMP2015给出的微博数据，1000万条）进行OOV(out-of-vocabulary)替换（替换为与词汇表最近的词）

>参考论文： [基于词矢量相似度的短文本分类](http://www.cnki.net/KCMS/detail/detail.aspx?QueryID=0&CurRec=1&recid=&filename=SDDX201412004&dbname=CJFDLAST2015&dbcode=CJFQ&pr=&urlid=&yx=&v=MDE1MzkxRnJDVVJMeWZiK1JyRnkvaFVieklOaW5QZHJHNEg5WE5yWTlGWUlSOGVYMUx1eFlTN0RoMVQzcVRyV00=)

代码中，需要设置LogisticRegression的参数
```python
clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
```

卷积神经网络
> 卷积神经网络在NLP中的使用多种多样，这里使用设置不同窗口大小的方法进行探索，即seq-CNN和Bow-CNN

>参考论文： [ (Johnson and Zhang, NAACL 2015) Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/pdf/1412.1058.pdf)

Seq-CNN  
由**one-hot编码**拼接而来
> 优点：词语之间顺序的得到保留
> 缺点：维度过大，容易造成维度灾难

Bow-CNN  
在**Seq-CNN**的基础上，进行降维
> 在确定窗口大小为n的情况，n之内的one-hot coding进行对应位数相加
优点：窗口内的语序信息丢失
缺点：窗口间的语序信息得到保留，维度得到降低


随机森林  
传统的**bagging融合模型**，这里**树的棵树**使用交叉验证得到，**树的深度**使用经验值：
> log(M)，其中M为总特征数

评价指标
> 准确率： sum(test_data_label == clf.predict(test)) / (1.0 * len(test_data_label))
