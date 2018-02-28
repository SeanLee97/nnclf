![PyTorch](https://raw.githubusercontent.com/SeanLee97/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

# nnclf

分本分类的方法有很多种，而且传统的机器学习方法就能有很好的效果，
神经网络的话在小数据量下是体现不出效果的，而且神经网络训练周期长。
分享这个项目就当作加深对卷积神经网络CNN以及递归神经网络RNN的理解吧。

**那么本项目主要有哪些亮点呢？**

    1. 支持batch_size，短文本一般长度不一，所以用PyTorch的话一般得要自己封装好输入向量
    看PyTorch文档时是不是在LSTM，GRU...这些地方会看到pack_padded_sequence(), pad_packed_sequence()呢？
    在这里就可以用到了
    2. 支持word2vec训练后的词向量表代替Embedding的权值。是不是有很多地方都说使用预训练的词向量效果会好？那究竟怎么用呢？
    在这里就可以用到了

**以上两点我都做了封装，看代码应该可以能理解，[使用方法](https://github.com/SeanLee97/nnclf/blob/master/Usage.ipynb)**


## 运行环境

* **Centos7**   当然其他linux, mac都可以，只要支持PyTorch的系统都可以，不过不支持windows
* **python3.6** 
* **PyTorch V0.2** 记得更新一下PyTorch，因为用到了lr_scheduler，新版才有

## 知识储备

* [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* 虽然有现成解决方案，但强烈推荐看一下word2vec原理，CBOW，Skip-gram，Hierarchical Softmax，Negative Sampling！！！[关于word2vec原理的博客](http://blog.csdn.net/itplus/article/details/37969519)

## 联系我
* 邮件(xmlee97#gmail.com, 把#换成@)
* weibo: [@捏明](http://weibo.com/littlelxm)

## 项目链接
[SVM, FastText, TextCNN, BiGRU, CNN-BiGRU在短文本分类上的对比](https://github.com/SeanLee97/short-text-classification#short-text-classification)
