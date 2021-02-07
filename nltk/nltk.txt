import nltk
from nltk.corpus import wordnet as wn

# 查询一个词所在的所有词集（synsets
wn.synsets('food')[2].definition()

# 查询一个同义词集的定义
wn.synset('apple.n.01').definition()

# 查询词语一个词义的例子
wn.synset('dog.n.01').examples()

# 查询词语某种词性所在的同义词集合
wn.synsets('dog',pos=wn.NOUN)

# 查询一个同义词集中的所有词
wn.synset('dog.n.01').lemma_names( )

# 输出词集和词的配对——词条（lemma）
wn.synset('dog.n.01').lemmas( )

# 利用词条查询反义词
good = wn.synset('good.a.01')
good.lemmas()[0].antonyms()

# 查询两个词之间的语义相似度
# path_similarity函数，值从0-1，越大表示相似度越高

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
dog.path_similarity(cat)

# 值得注意的是，名词和动词被组织成了完整的层次式分类体系，形容词和副词没有被组织成分类体系，所以不能用path_distance。 
# 形容词和副词最有用的关系是similar to

# 蕴含：entailments()
# 以上的分析多是针对名词，对于动词，也存在关系。这里只有列出了一种蕴含的关系： 
# entailments()方法，同样由一个词集调用：

wn.synset('walk.v.01').entailments() #走路蕴含着抬脚

# 反义词：antonyms()
# 由一个词条调用：wn.lemma(‘supply.n.02.supply’)

wn.lemma('supply.n.02.supply').antonyms()














