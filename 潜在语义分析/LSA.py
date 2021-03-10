import numpy as np
import math
import re
import nltk
from nltk.corpus import wordnet
# 1获得所有句子 完成
# 2 将句子分隔成单词 完成
# 3 将同一单词的不同形式都归结成一个单词 完成
# 4 构建 文本-单词 矩阵
from nltk.stem import WordNetLemmatizer
class LSA:
    def __init__(self,word_sentence,sentence_index,index_sentence,word_index, index_word,k):
        self.word_sentence=np.array(word_sentence)
        self.sentence_index=sentence_index
        self.index_sentence=index_sentence
        self.word_index3es=word_index
        self.index_word=index_word
        self.k=k
        self.u, self.s1, self.vt, self.r = self.cal_value_and_vector()

    def cal_value_and_vector(self):
        # 返回特征值与特征向量
        u, s, vt = np.linalg.svd(self.word_sentence)
        s1 = np.array([[0] * self.word_sentence.shape[1]] * self.word_sentence.shape[0])
        k = 0
        # print(s)
        # print(s.shape)
        for i in range(self.word_sentence.shape[0]):
            for j in range(self.word_sentence.shape[1]):
                if i == j:
                    s1[i][j] = s[k]
                    k += 1
        return u, s1, vt, s.shape[0]

    def cal(self):
        # 返回单词- 话题 话题-文本
        uk = self.u[:, 0:self.k]
        s1k = self.s1[0:self.k, 0:self.k]
        vk = self.vt.T[:, 0:self.k]
        vkt = vk.T
        return uk,np.dot(s1k, vkt)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def get_sentences():
    #将一篇文章分割成句子返回
    # 用来分割句子的分隔符
    pattern = r'\.|\?|!'
    # 打开一篇英语文章
    f = open('article.txt', 'r') #在txt中将整个文章挤成一个句子
    list = f.readlines()  #读取
    result_list = re.split(pattern, list[0]) #用分割符将英语文章分割成一个个句子
    result_list.remove("") #去除空字符
    # for i in range(len(result_list)):
    #     print(result_list[i])
    return  result_list
def get_words(result_list):
    #返回所有句子中包含的单词 以字典形式 单词: 出现该单词的句子数目
    # texts={}
    # for i in range(len(result_list)):
    #    texts[result_list[i]] = i
    # print(texts)
    pattern = r';|,|\s'
    words={}
    wnl = WordNetLemmatizer()
    for i in range(len(result_list)):
        #获取某条句子
       text=result_list[i]
        #分割成单词
       list=re.split(pattern,text)
       if "" in list:
          list.remove("")
       #获取单词的标签
       tagged_sent = nltk.pos_tag(list)
       sw=[] #这个句子的单词保证不重复
       for (word,token) in tagged_sent: #单词 标签
          w=word.lower() #将所有大写字母转为小写字母
          get_wordnet_pos_value= get_wordnet_pos(token)
          if get_wordnet_pos_value!=None: # 如果这个单词是名词 动词 形容词 副词
              #将这个单词的各种变形转换为原始形式
              w=wnl.lemmatize(w, get_wordnet_pos_value)
          if w not in sw:
            sw.append(w)
       for w in sw:
            if w not in words:
               words[w]=1
            else:
                words[w] +=1
   # print(words)
    return words
def create_word_sentence(sentences,words):
    #构建单词，文本矩阵
    #句子下标矩阵
    sentence_index={}
    index_sentence=[]
    #单词下标矩阵
    word_index={}
    index_word = []
    wnl = WordNetLemmatizer()
    for i in range(len(sentences)):
        index_sentence.append(sentences[i])
        sentence_index[sentences[i]]=i
    i=0
   # print(words)
    for k,v in words.items():
        word_index[k] = i
        index_word.append(k)
        i+=1
    word_sentence=[[0.0]*len(sentences)]*len(words)
    pattern = r';|,|!|\s'
    for i in range(len(index_sentence)):
        text = index_sentence[i]
        list = re.split(pattern, text)
        if "" in list:
            list.remove("")
        sum=len(list) #这个句子中一共有多少个单词
        tagged_sent = nltk.pos_tag(list)
        sw = {}  # 这个句子的单词保证不重复
        for (word, token) in tagged_sent:  # 单词 标签
            w = word.lower()  # 将所有大写字母转为小写字母
            get_wordnet_pos_value = get_wordnet_pos(token)
            if get_wordnet_pos_value != None:  # 如果这个单词是名词 动词 形容词 副词
                # 将这个单词的各种变形转换为原始形式
                w = wnl.lemmatize(w, get_wordnet_pos_value)
            if w not in sw:
                sw[w]=1
            else:
                sw[w] += 1
        for word,num in sw.items():
            v=num/sum*math.log(len(index_sentence)/words[word])
            word_sentence[word_index[word]][i]=v
    return  word_sentence,sentence_index,index_sentence,word_index, index_word
sentences=get_sentences()
words=get_words(sentences)
word_sentence,sentence_index,index_sentence,word_index, index_word=create_word_sentence(sentences,words)
print(word_sentence)
lsa=LSA(word_sentence,sentence_index,index_sentence,word_index, index_word,5)
X1,X2=lsa.cal()
print(X1)
print(X2)