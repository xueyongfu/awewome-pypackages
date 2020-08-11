import requests
import time
import os


_url = 'http://127.0.0.1:%d%s'%(6060, '/api/test')
_headers = {'Content-Type': 'application/x-www-form-urlencoded'}

def pos_ner_server():
    data = {'sentence':'今天是个好天气', 'orderId':'4434385932'}
    res = requests.post(_url, data, headers=_headers, verify=False)
    print(res.content.decode())


if __name__ == '__main__':
    pos_ner_server()