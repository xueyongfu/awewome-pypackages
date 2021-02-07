class A:
    def __init__(self):
        self.t1()   #如果将此行注销掉，会报错，self.num会识别不了
    
    def t1(self):
        self.num = 111

    def t2(self):
        print(self.num)
    
a = A()
a.t2()

# Super的作用

# 如果子类(Puple)继承父类(Person)不做初始化，那么会自动继承父类(Person)属性name。
# 如果子类(Puple_Init)继承父类(Person)做了初始化，且不调用super初始化父类构造函数，那么子类(Puple_Init)不会自动继承父类的属性(name)。
# 如果子类(Puple_super)继承父类(Person)做了初始化，且调用了super初始化了父类的构造函数，那么子类(Puple_Super)也会继承父类的(name)属性。
class A:
    def __init__(self):
        print('A')
        
class B(A):
    def __init__(self):
        print('B')
        super().__init__()

class C(A):
    def __init__(self):
        print('C')
        super().__init__()

class D(A):
    def __init__(self):
        print('D')
        super().__init__()
        
class E(B, C):
    def __init__(self):
        print('E')
        super().__init__()


class F(C, D):
    def __init__(self):
        print('F')
        super().__init__()

class G(E, F):
    def __init__(self):
        print('G')
        super().__init__()

# 输出当前时间

from datetime import datetime
now = datetime.now()
now

# 输出当前时间的年、月、日
# now.year, now.month, now.day

# 计算两个时间点间的时间差

datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
# 输出#即926天，56700秒

# 时间相加

from datetime import timedelta

start = datetime(2011, 1, 7)
start + timedelta(12) #默认是天数，等同于timedelta(days=12)

# 日期转字符串

from datetime import datetime
now = datetime.now()
print(now.strftime('%a, %b %d %H:%M'))

# 读取数据，然后清空数据并写入

fx = open(history_best_score_path,'r+')
fx.seek(0,0)
fx.write(self.his_best_score)

# 文件读取方式

read()   #将文本文件所有行读到一个字符串中。
readline()    #是一行一行的读
readlines()      #是将文本文件中所有行读到一个list中，文本文件每一行是list的一个元素。优点：readline()可以在读行过程中跳过特定行。

# 大文件读取

# 方式1：分块读取
# 将大文件分割成若干小文件处理，处理完每个小文件后释放该部分内存。这里用了iter 和 yield
def read_in_chunks(filePath, chunk_size=1024*1024):
    file_object = open(filePath)
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data
        
if __name__ == "__main__":
    filePath = './path/filename'
    for chunk in read_in_chunks(filePath):
        process(chunk)  # <do something with chunk>

# 方式2：for line in f 方式
# for line in f: 文件对象f视为一个迭代器，会自动的采用缓冲IO和内存管理，所以你不必担心大文件。 
# 一般的使用方法是:
with open(file_path) as f:
　　for line in f:
        (1)process(line)  
        (2)将处理后的结果存储起来
        (3)将这部分数据drop掉

# 二进制文件读取

path = '/home/xyf/桌面/seq2seq/fairseq/data-bin/test.de-en.de.bin'
import struct

with open(path,'rb') as f:
    i = 0
    while i < 20:
        i += 1
        data = f.read(4)
        p = struct.unpack('<i',data)

# 注释:读取二进制文件相当于一个解码过程,不知道格式是没办法读取的,或者读取的内容是错误的

# 列表函数与方法

序号	函数
1	len(list)列表元素个数
2	max(list)返回列表元素最大值
3	min(list)返回列表元素最小值
4	list(seq)将元组转换为列表
Python包含以下方法:



序号	方法
1	    list.append(obj)      在列表末尾添加新的对象
2	    list.count(obj)       统计某个元素在列表中出现的次数
3	    list.extend(seq)      在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
4	    list.index(obj)       从列表中找出某个值第一个匹配项的索引位置
5	    list.insert(index, obj)   将对象插入列表
6	    list.pop([index=-1])  移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
7	    list.remove(obj)      移除列表中某个值的第一个匹配项
8	    list.reverse()        反向列表中元素
9	    list.sort( key=None, reverse=False)   对原列表进行排序
10	    list.clear()         清空列表
11	    list.copy()          复制列表

# 集合内置方法完整列表

方法	                         描述
add()	                        为集合添加元素
clear()	                        移除集合中的所有元素
copy()	                        拷贝一个集合
difference()	                返回多个集合的差集
difference_update()	            移除集合中的元素，该元素在指定的集合也存在。
discard()	                    删除集合中指定的元素
intersection()	                返回集合的交集
intersection_update()	        返回集合的交集。
isdisjoint()	                判断两个集合是否包含相同的元素，如果没有返回 True，否则返回 False。
issubset()	                    判断指定集合是否为该方法参数集合的子集。
issuperset()	                判断该方法的参数集合是否为指定集合的子集
pop()	                        随机移除元素
remove()	                    移除指定元素
symmetric_difference()	        返回两个集合中不重复的元素集合。
symmetric_difference_update()	移除当前集合中在另外一个指定集合相同的元素，并将另外一个指定集合中不同的元素插入到当前集合中。
union()	                        返回两个集合的并集
update()	                    给集合添加元素

# 参数*//**

dic = {'a':1,'b':2}

def a(a, b, c=0, d=0):
    print(a)
    print(b)

a(**dic)
















