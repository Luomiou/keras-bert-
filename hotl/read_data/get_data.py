
import os
import sys
import re
import jieba

syspath=sys.path[1]
pos_data_path=syspath+'\hotl\data\pos'
neg_data_path=syspath+'\hotl\data\\neg'

def get_data(path,filename):
    # 获取path文件夹下的文件名，返回list
    filenames=os.listdir(path)
    # fname=open(syspath+'\hotl\data\\'+filename+'.txt','a',encoding='UTF-8')
    filelist=[]
    for file in filenames:
        newDir = path+'\\'+file
        if os.path.isfile(newDir):
            if os.path.splitext(newDir)[1] == '.txt':
                f=open(newDir,encoding='UTF-8')
                str=''
                for line in f.readlines():
                    str += line.strip()
                # fname.writelines(str+'\n')
                str=delete_fuhao(str)
                filelist.append(str)

    return filelist

def delete_fuhao(stri):
    strings = re.sub("[\s+\.\!\/_,$%^*(+\"\'+-=]+|[+——！>＞＜，∞?。"
                     + "‘？、~@#￥>%……&*（）::【】“”""““””～·《》"
                     + "〔〕」)」×→→\％＋：]+|[\!\%\[\]\,\。]", "", str(stri))
    result=''
    if len(stri)!=0:
        seg_jieba_list=jieba.cut(strings)
        for w in seg_jieba_list:
            result +=' '+w
    return result.strip()





















