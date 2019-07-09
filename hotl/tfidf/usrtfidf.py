
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from hotl.read_data.get_data import pos_data_path, get_data, neg_data_path

import numpy as np

def tfidf(count_list):
    vectorizer=CountVectorizer()
    x=vectorizer.fit_transform(count_list)
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(x)
    result=tfidf.toarray()
    return result

def split_train_test(train_list):
    y=np.concatenate((np.ones(3000,dtype=int),np.zeros(3000,dtype=int)))
    print(y)
    train_x,train_y,test_x,test_y=train_test_split(train_list,y,test_size=0.1)

    # lr=LogisticRegression(penalty='l2',dual=False,tol=0.0001,C=1,fit_intercept=True)
    # lr.fit(train_list,y)
    # # score=lr.score(test_x,test_y)
    # # print(score)







poslist=get_data(pos_data_path,'pos')
neglist=get_data(neg_data_path,'neg')
count_list=poslist+neglist
train_list=tfidf(count_list)
split_train_test(train_list)




