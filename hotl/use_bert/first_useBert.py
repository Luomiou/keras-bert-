# encoding = utf - 8


import codecs
import os
import sys

import numpy as np
import yaml
from keras import Input, Model, losses, Sequential
from keras.activations import relu, sigmoid
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint

syspath = sys.path[1]+'/hotl'
os.chdir(syspath)
# 注意！！！！特别注意！！！此处要慎重，以下内容是Bert-Pre-training
config_path = 'D:\Pycharm\post\hotl\chinese_L-12_H-768_A-12\\bert_config.json'# 加载配置文件
checkpoint_path = 'D:\Pycharm\post\hotl\chinese_L-12_H-768_A-12\\bert_model.ckpt'
dict_path = 'D:\Pycharm\post\hotl\chinese_L-12_H-768_A-12\\vocab.txt'
#  以上是模型的加载，要慎重，一定要慎重！！！！！

maxlen=100# 句子的最大长度，padding要用的

def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:将文件中字进行编码
    '''
    # 将bert模型中的 字 进行编码
    # 目的是 喂入模型  的是  这些编码，不是汉字
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

class OurTokenizer(Tokenizer):
    '''
    关键在  Tokenizer 这个类，要实现这个类中的方法，其实不实现也是可以的
    目的是 扩充 vocab.txt文件的
    '''
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


def get_data():
    '''
    读取数据的函数
    :return: list  类型的 数据
    '''
    pos = []
    neg = []
    with codecs.open('./data/pos.txt','r','utf-8') as reader:
        for line in reader:
            pos.append(line.strip())
    with codecs.open('./data/neg.txt','r','utf-8') as reader:
        for line in reader:
            neg.append(line.strip())
    return pos,neg


# 得到编码
def get_encode(pos,neg,token_dict):
    '''
    :param pos:第一类文本数据
    :param neg:第二类文本数据
    :param token_dict:编码字典
    :return:[X1,X2]，其中X1是经过编码后的集合，X2表示第一句和第二句的位置，记录的是位置信息
    '''
    all_data = pos + neg
    tokenizer = OurTokenizer(token_dict)
    X1 = []
    X2 = []
    for line in all_data:
        # tokenizer.encode(first,second, maxlen)
        # 第一句和第二句，最大的长度，
        # 本数据集是  都是按照第一句，即一行数据即是一句，也就是第一句
        # 返回的x1,是经过编码过后得到，纯整数集合
        # 返回的x2,源码中segment_ids，表示区分第一句和第二句的位置。结果为：[0]*first_len+[1]*sencond_len
        # 本数据集中，全是以字来分割的。
        # line_list = line.split('.')
        # for i in line_list:
        #     print(i)
        x1,x2 = tokenizer.encode(first=line)
        # print(line,'\n')
        # print(x1,'\n',len(x1),'\n',x2,'\n',len(x2))
        # break
        X1.append(x1)
        X2.append(x2)
    # 利用Keras API进行对数据集  补齐  操作。
    # 与word2vec没什么区别
    X1 = sequence.pad_sequences(X1,maxlen=maxlen,padding='post',truncating='post')
    X2 = sequence.pad_sequences(X2,maxlen=maxlen,padding='post',truncating='post')
    return [X1,X2]


def build_bert_model(X1,X2):
    '''
    :param X1:经过编码过后的集合
    :param X2:经过编码过后的位置集合
    :return:模型
    '''

    #  ！！！！！！ 非常重要的！！！非常重要的！！！非常重要的！！！
    # 加载  Google 训练好的模型bert 就一句话，非常完美prefect
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # config_path 是Bert模型的参数，checkpoint_path 是Bert模型的最新点，即训练的最新结果
    # 特别注意的是  加载Bert的路径 问题，
    # 注：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip，
    #     下载完之后，解压得到4个文件，直接放到 项目的路径下，要写上绝对路径，以防出现问题。
    # 安装 keras-bert：pip install keras-bert
    wordvec = bert_model.predict([X1,X2])
    return wordvec

def build_model():
    model =Sequential()
    model.add(Dense(128,activation=relu))
    model.add(Dense(1,activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(1e-5), metrics=['accuracy'])
    
    return model

def train(wordvec,y):
    model = build_model()
    model.fit(wordvec,y,batch_size=32,epochs=10,validation_split=0.2)
    model.summary()
    yaml_string = model.to_yaml()
    with open('test_keras_bert.yml', 'w') as f:
        f.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('test_keras_bert.h5')

if __name__ =='__main__':
    pos,neg = get_data()
    token_dict = get_token_dict(dict_path)
    # get_encode()
    [X1,X2] = get_encode(pos,neg,token_dict)
    wordvec = build_bert_model(X1,X2)
    # 标签类，其中选取3000个积极的文本和3000个消极的文本，将积极的记为1，将消极的记为0
    y = np.concatenate((np.ones(3000, dtype=int), np.zeros(3000, dtype=int)))
    # y = keras.utils.to_categorical(y,num_classes=2)
    # p = Dense(2, activation='sigmoid')(x)
    train(wordvec,y)


