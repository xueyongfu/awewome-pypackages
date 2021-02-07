from py2neo import Graph, Node, Relationship

graph = Graph("http://172.18.1.91:7474", username="neo4j", password="123456")

#创建节点和关系,当创建关系时,如何不存在对应的节点,会自动创建
a = Node("Person", name="Alice")
b = Node("Person", name="Bob")
ab = Relationship(a, "KNOWS", b)

# 当图数据库中存在节点和关系时,create会创建重复的节点关系
# 不重复节点关系的创建使用merge
# graph.create(a)
# graph.create(b)
graph.merge(ab)

#在创建好node后，我们可以有很多操作
node = Node("地点", name="上海", attri='直辖市', belong='中国', other='其他')

# name属性是节点名字,没有name属性便会没有节点名字,系统会随机给一个id
# node = Node("地点", nam="上海", attri='直辖市', belong='中国', other='其他')
# graph.create(node)

#获取key对应的property
x=node['attri'] 
print(x)

#设置key键对应的value，如果value是None就移除这个property
node['belong'] = '世界'

#也可以专门删除某个property
del node['other']

#返回node里面property的个数
print(len(node))

#返回所有和这个节点有关的label
labels=node.labels
print(labels)

#删除某个label
# node.labels.remove(labelname)

#将node的所有property以dictionary的形式返回
dict(node)

#对于关系的操作

#创建Relationship
# Relationship`(*start_node*, *type*, *end_node*, ***properties*)

ab = Relationship(a, "KNOWS", b, attri='类别')

#返回Relationship的property
print(ab['attri'])

#删除某个property
del ab['attri']

#将relationship的所有property以dictionary的形式返回
dict(ab)


# 子图是节点和关系不可变的集合,我们可以通过set operator来结合，参数可以是独立的node或relationships
# 节点和关系都可以看做一个子图

# subgraph | other | ...      结合这些subgraphs
# subgraph & other & ...   相交这些subgraphs
# subgraph - other - ...     不同关系
# #比如我们前面创建的ab关系
# s = ab | ac

from py2neo import Node, Relationship

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
s = a | b | r

# 通过 nodes() 和 relationships() 方法获取所有的 Node 和 Relationship
print(s.nodes())
print(s.relationships())

# merge()函数
#注意: 图数据库存在同名节点,但是有不同的属性,那么使用merge()会新创建一个同名节点,除非名字属性均相同.

graph.merge(subgraph=subgraph)

 
list(graph.run("MATCH (a:Person {name:'Alice'}) RETURN a"))

#查询节点
grapg.find()
grapg.find_one()


#查询关系
grapg.match()
grapg.match_one()

# NodeMatcher是为更好的查询节点，支持更多的查询条件，比graph更友好
from py2neo import NodeSelector
selector = NodeSelector(graph)
list(selector.select("Person", name="Alice"))

# where()函数中可以添加多种条件
list(selector.select("Person").where("_.name =~ 'J.*'",  "1960 <= _.born < 1970"))  #_表示节点

# order_by()进行排序
persons = selector.select('Person').order_by('_.age')  #按照年龄进行排序
print(list(persons))

# selector的属性总结
first()返回单个节点
limit(amount)返回底部节点的限值条数
skip(amount)返回顶部节点的限值条数
order_by(*fields)排序
where(*conditions, **properties)筛选条件

# Walkable Types是一个拥有遍历功能的子图。最简单的构造就是把一些子图合并起来

from py2neo import Node, Relationship

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
c = Node('Person', name='Mike')
ab = Relationship(a, "KNOWS", b)
ac = Relationship(a, "KNOWS", c)
w = ab + Relationship(b, "LIKES", c) + ac
print(w)

# 我们可以使用walk方法实现遍历
from py2neo import walk

for item in walk(w):
    print(item)

# 利用 start_node()、end_node()、nodes()、relationships() 方法来获取起始 Node、终止 Node、所有 Node 和 Relationship
print(w.start_node())
print(w.end_node())
print(w.nodes())
print(w.relationships())

# push用来更新图数据库节点
# 先查询出来,再更新节点

node = graph.find_one(label='Person')
node['age'] = 18
graph.push(node)
print(graph.find_one(label='Person'))

# update()函数并不能更新图数据库节点,可以用来批量更新节点属性
a = Node()
data = {'name': 'Amy','age': 21}
a.update(data)

# delete(subgraph) 删除节点、关系或子图 , 删除关系时,关系的节点也会删除
# delete_all() 删除数据库所有的节点和关系

a = Node("Person", name="Alice")
b = Node("Person", name="Bob")
ab = Relationship(a, "KNOWS", b)
graph.create(ab)

node = graph.find_one(label='Person')
relationship = graph.match_one(node,rel_type='KNOWS')
graph.delete(relationship)
















