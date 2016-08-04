# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:25:56 2016

@author: Administrator
"""
import sys
import numpy as np
import jieba
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.externals import joblib

#label in line[0:1] and content in line[1:]
#stopwords = {}.fromkeys([line.rstrip() for line in open(r'F:\dict\stopword.txt')])

def get_data(file):
    label=[]
    content=[]
    f = open(file,'r', encoding='utf-8', errors='ignore')
    for line in f :
        if line :
            line = line.replace('"','')
            label.append(line[0:1])        
            content.append(line[1:])
    print("Finish loading, size: ", len(label))
    return content, np.array(label)
    
"""
def get_old_data(file):
    label=[]
    content=[]
    f = open(file,'r', encoding='gbk', errors='ignore')
    for line in f :
        if line :
            #line.replace('"','')
            cut = line.split('\t')
            label.append(cut[0])        
            content.append(cut[1])
    print("Finish loading, size: ", len(label))
    return content, np.array(label)
""" 

#分词+向量化，返回矩阵
def preprocess(text):
    cutlist = []
    jieba.load_userdict(r"F:\dict\anti-spam-user-dict.txt")
    for line in text:
        cw = jieba.cut(line)
        cutlist.append(' '.join(cw))
    print("Finish Cutting Words, size", len(cutlist))
    corpus = np.array([line.strip() for line in cutlist])
   # matrix = CountVectorizer().fit_transform(corpus)
    return corpus

def data_process(matrix, label, test_size=0.5):
    x_train, x_test, y_train, y_test = train_test_split(matrix, label, test_size=test_size, random_state=42)
    print("Split data: test size " + str(test_size))
    print("TrainingSet size: ", len(y_train))
    return x_train, x_test, y_train, y_test

def model_train(x, y, chi2_num=50):
    mod = Pipeline([('vect', CountVectorizer()),
                    ('feature_sel', SelectPercentile(chi2, chi2_num)),
                    #('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance'))
                    #('NB',MultinomialNB())
                    #('RF', RandomForestClassifier(n_estimators=30))
                    ('LR', LogisticRegression(penalty='l1', C=1.5))
                    #('SVM', svm.SVC(kernel='linear'))
                    ])
    t0 = time()
    clf = mod.fit(x, y)
    t1 = time()
    print("Finish, FIT time: %0.3fs" % (t1-t0))
    joblib.dump(clf, r'F:\model\LR_clean.pkl')
    return clf

def model_test(clf, x_test, y_test):
    print("Predict data")
    t0 = time()
    pred = clf.predict(x_test)
    print("Finish, predict time: %0.3fs" % (time()-t0))
    print(metrics.classification_report(y_test, pred))
    print(metrics.confusion_matrix(y_test, pred))
    

if __name__ == '__main__':
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    #file = sys.argv[1]
    file = r'F:\test\test_data_clean.txt'
    print("Begin Trainnign model, load data...")
    x, y = get_data(file)
    print("Pre-processing data, cut words...")
    cor = preprocess(x)
    print("Split data...")
    x1,x2,y1,y2 = data_process(cor,y,0.1)
    print("Start training model:")
    clf = model_train(x1,y1,70)
    print("model test:")
    model_test(clf,x2,y2)




