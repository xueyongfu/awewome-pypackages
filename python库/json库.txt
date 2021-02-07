import json

# 包含中文的字典使用json.dumps()后,中文不能正常显示
print(json.dumps({'姓名':'小明'}))
print(json.dumps({'姓名':'小明'}, ensure_ascii=False))
















