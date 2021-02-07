# 中文分词

import synonyms
synonyms.seg("中文近义词工具包")

# 近义词

import synonyms
print("人脸:", (synonyms.nearby("人脸")))
print("识别:", (synonyms.nearby("识别")))

# 返回两个列表,列表一是近义词们,按照距离的长度由近及远排列，列表二是中对应位置的词的距离的分数，分数在(0-1)区间内，越接近于1，代表越相近。


# 获取近义词2
synonyms.display("飞机")

# 两个句子的相似度比较
# 旗帜引领方向 vs 道路决定命运: 0.429
# 旗帜引领方向 vs 旗帜指引道路: 0.93
# 发生历史性变革 vs 发生历史性变革: 1.0

sen1 = "旗帜引领方向"
sen2 = "旗帜指引道路"
r = synonyms.compare(sen1, sen2, seg=True)  #seg表示s
print(r)

# 获得一个词语的向量，该向量为numpy的array，当该词语是未登录词时，抛出 KeyError异常。

synonyms.v("飞机")

# 获得一个分词后句子的向量，向量以BoW方式组成

# sentence: 句子是分词后通过空格联合起来
# ignore: 是否忽略OOV，False时，随机生成一个向量
sentence = '我很开心'
synonyms.sv(sentence, ignore=False)










