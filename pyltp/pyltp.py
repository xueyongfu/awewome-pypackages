#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
from pyltp import Segmentor, Postagger, Parser ,SementicRoleLabeller,NamedEntityRecognizer
import numpy as np


# In[11]:


class nlpLtp:
    
    def __init__(self):
        MODELDIR = '/home/xyf/models/chinese/ltp_model'

        #系统切词
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(MODELDIR, "cws.model"))

        self.postagger = Postagger()
        self.postagger.load(os.path.join(MODELDIR, "pos.model"))

        self.namedentityrecognizer = NamedEntityRecognizer()
        self.namedentityrecognizer.load(os.path.join(MODELDIR, "ner.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(MODELDIR, "parser.model"))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(MODELDIR, "pisrl.model"))

        self.parse_dict =  {"SBV":"主谓关系", "VOB":"动宾关系", "IOB":"间宾关系", "FOB":"前置宾语", "DBL":"兼语",
                           "ATT":"定中关系", "ADV":"状中关系", "CMP":"动补关系", "POB":"介宾关系", "LAD":"左附加关系",
                           "RAD":"右附加关系", "IS":"独立结构","COO":"并列关系", "HED":"核心关系", "WP":"标点"}

    def sent_segment(self, sentence):
        words_ltp = self.segmentor.segment(sentence)
        words_list = [w for w in words_ltp]
        return words_list

    def sent_pos(self, sentence):
        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        return postags

    def sent_ner(self, sentence):
        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        netags = self.namedentityrecognizer.recognize(words, postags)
        return netags

    def sent_syntax(self, sentence):
        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        parsing = self.parser.parse(words, postags)
        syntax = "  ".join("%d:%s" % (pars.head, pars.relation) for pars in parsing)
        return parsing, syntax

    def sent_syntax_self(self,sentence):
        print ('原文本：' + sentence)
        words = self.sent_segment(sentence)
        print('分词结果：' + str(words))
        postags = self.sent_pos(sentence)
        print('词性标注结果：' + str([a for a in postags]))
        parsing = self.parser.parse(words, postags)
        parsing_a = "  ".join("%d:%s" % (pars.head, pars.relation) for pars in parsing)
        print('句法分析结果：' + parsing_a)

        parsing_b = zip(words, parsing)
        for par in parsing_b:
            if par[1].relation in ['WP', 'HED']:
                print('"'+par[0]+'"是'+self.parse_dict[par[1].relation], end=',  ')
            else:
                print('"'+par[0]+'"'+'与'+'"'+words[par[1].head-1]+'"'+'：'+self.parse_dict[par[1].relation], end=',  ')

    def sent_role(self, sentence):
        words = self.sent_segment(sentence)
        postags = self.sent_pos(sentence)
        parsing = self.parser.parse(words, postags)

        roles = self.labeller.label(words, postags, parsing)
        for role in roles:  #roles是谓词
            print(role.index, "".join(
                ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
            
            for arg in role.arguments:
                if arg.name == 'A1':
                    words_list=words[arg.range.start:arg.range.end+1]
                    print(''.join(words_list))

# 构造一个对象
nlpltp = nlpLtp()


# In[12]:


print(nlpltp.sent_segment('客户来电反映之前区营业厅办理卡的时候工作人员有为客户参与AA421066_全国不限量68包打98（6个月）和办理ACBZ14195 新爱家88（V2.0），13408470690 全球通 营销执行（电子渠道营销） 2039003395367  2019-06-28 12:09:45 ane130007(成都温江分公司龙翔通讯永兴路延迟结酬至202001李川川)  营业厅 无 电子化渠道营销执行 0.00 身份证件 '))


# In[12]:


print(list(nlpltp.sent_pos('李克强总理今天来我家了,我感到非常荣幸')))


# In[14]:


print(list(nlpltp.sent_ner('【手机】市民反映：浦东新区曹路镇顾曹公路市场路有乱设摊，市民未提供具体地址，具体设摊时间，诉求：请管理部门核实后尽快协调制止。')))


# In[14]:


import json
a={
    "segment":["李克强", "总理", "今天", "来", "我家", "了", ",", "我", "感到", "非常", "荣幸"],
    "pos":["nh", "n", "nt", "v", "n", "u", "wp", "r", "v", "d", "a"],
    "ner":["S-Nh", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
json.dumps(a)


# In[4]:


print(nlpltp.sent_syntax('李克强总理今天来我家了,我感到非常荣幸'))


# In[25]:


print(nlpltp.sent_syntax_self('近日，一条男子高铁吃泡面被女乘客怒怼的视频引发热议'))


# In[21]:


# 语义角色分析
# 核心的语义角色为 A0-5 六种，A0 通常表示动作的施事，A1通常表示动作的影响等，A2-5 根据谓语动词不同会有不同的语义含义。其余的15个语义角色为附加语义角色，如LOC 表示地点，TMP 表示时间等。附加语义角色列表见LTP官方说明文档

print(nlpltp.sent_role('今天上午我想看恐龙来了'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




