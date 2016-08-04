
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:37:38 2016
this is for the single test.
@author: Tomi
"""

import jieba
from sklearn.externals import joblib
from time import time
import numpy as np

def cutwords(text):
    jieba.load_userdict(r"F:\dict\anti-spam-user-dict.txt")
    cw = jieba.cut(text)
    s = ' '.join(cw)
    return s

def single_predict(text, label, clf_file = r'F:\model\LR_clean.pkl'):
    label_list = ["0-正常","1-低俗（宽松）","2-低俗（一般）","3-低俗（禁言）","4-语言暴力及谩骂","5-涉政","6-地域攻击","7-个人隐私"]
    clf = joblib.load(clf_file)
    s = cutwords(text)
    X = np.array([s])
    t2 = time()    
    #s = clf.predict(X)
    pred = clf.predict_proba(X)
    t3 = time()
    pre_label_list = pred[0].tolist()
    print("Single Predict: %0.3fms" % ((t3-t2)*1000))
    print("content : ", s)
    print("True label: ", label_list[int(label)])
    print("Predict label: ", label_list[pre_label_list.index(max(pre_label_list))])
    print("Acc: %0.2f" % np.max(pred))
    return

if __name__ == '__main__':
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    #file = sys.argv[1]
    s = input("single predict text here: ")
    label = input("real label: ")
    print('\n')
    single_predict(s,label)