import re

# | 或(两边不能有空格)

re.sub('\d+|,','','12,sddf,,d123ds')

# re.match
# re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。
print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配

# re.search方法
# re.search 扫描整个字符串并返回"第一个"成功的匹配。
print(re.search('www', 'www.runoob.com').span())     # 在起始位置匹配
print(re.search('com', 'www.runoob.com').span())         # 不在起始位置匹配

# re.sub 
# re.sub用于替换字符串中的匹配项。
# 语法：re.sub(pattern, repl, string, count=0, flags=0)    count:模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
phone = "2004-959-559 # 这是一个国外电话号码"
num = re.sub(r'#.*$', "", phone)
num

# re.compile 函数
# compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。
pattern = re.compile(r'\d+')                    # 用于匹配至少一个数字
m = pattern.match('one12twothree34four')        # 查找头部，没有匹配
m

m = pattern.search('one12twothree34four')        # 查找头部，没有匹配
m

# findall()函数
# 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表。
# 注意： match 和 search 是匹配一次 findall 匹配所有。
# 语法:findall(string[, pos[, endpos]])
    # 参数：
    # string : 待匹配的字符串。
    # pos : 可选参数，指定字符串的起始位置，默认为 0。
    # endpos : 可选参数，指定字符串的结束位置，默认为字符串的长度。

# 方式一:
pattern = re.compile(r'\d+')   
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
print(result1)
print(result2)

# 方式二
result3 = re.findall(r'\d+', 'runoob 123 google 456')
print(result3)

# 字符串分割

import re
text = '我很开心。哈哈？'
print(re.split('。|！|？',text))


#(...)
#re.findall与()结合使用,搜索的结果是()内的内容
#re.match或者re.search与()结合使用,返回字符串中的内容
string = '33sjf33djfj'
print(re.findall('\d+(\w*)', string))
print(re.match('\d+(\w)', string).group())

# 匹配由单个空格分隔的任意单词对，也就是姓和名
s = "Han meimei, Li lei, Zhan san, Li si"
print (re.findall(r'([A-Za-z]+) ([A-Za-z]+)',s))

# 匹配由单个逗号和单个空白符分隔的任何单词和单个字母,如姓氏的首字母
s = "yu, Guan  bei, Liu  fei, Zhang"
print(re.findall(r'([a-zA-Z]+),\s([a-zA-Z])',s))

# ()后面接上+,表示可以匹配一组或者多组
s = """street 1: 1180   Bordeaux Dr ive, street 1: 3120 De la Cruz Boulevard"""
print(re.findall(r'\d+( +[a-zA-Z]+)+',s))
print(re.search(r'\d+( +[a-zA-Z]+)+',s).group())

#在正则表达式里面加上?来使用非贪婪模式,默认是贪婪模式

a = '中华路中行路哈哈'
print(re.match(r'.*路',a).group())
print(re.match(r'.*?路',a).group())










































