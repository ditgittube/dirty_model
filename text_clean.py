# -*- coding: utf-8 -*-
"""
Test clean

Created on Wed Aug  3 14:27:48 2016

@author: Administrator
"""

import jieba

stopwords = {}.fromkeys([line.rstrip() for line in open(r'F:\dict\stopword.txt','r',encoding='utf-8', errors='ignore')])
jieba.load_userdict(r"F:\dict\anti-spam-user-dict.txt")

def text_clean(filename, save):
    cutlist=[]
    for line in open(filename, 'r', encoding='utf-8', errors='ignore'):
        line = line.replace("\"","")
        ls = list(jieba.cut(line))
        for w in ls:
            if w in stopwords:
                ls.remove(w)
                #print(w)
        cutlist.append(''.join(ls))
        print(len(cutlist))
    f = open(save,'w', encoding='utf-8', errors='ignore')
    f.writelines(cutlist)
    f.close()
    print("Finish cleaning")
    return

if __name__ == '__main__':
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    #file = sys.argv[1]
    file = input("please input filename(with complete diretory): ")
    save = file[:-4]+"_clean.txt"
    text_clean(file, save)
    
    