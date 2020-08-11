from aip.ocr import AipOcr

""" 你的 APPID AK SK """
APP_ID = '16904434'
API_KEY = 'BMAv0vyjbUSqgtqgfZa35bmp'
SECRET_KEY = 'BXXykWljc0QXrDyvbvTjqil04WVVTCib'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('images/2097109627.jpg')




if __name__ == '__main__':
    #手写文字识别
    text = client.handwriting(image)
    print(text)

    #识别文字保存
    path = 'output/detect_words.txt'
    with open(path,'w',encoding='utf-8') as f:
            tx_lists = text['words_result']
            for tx in tx_lists:
                    f.write(tx['words']+'\n')

