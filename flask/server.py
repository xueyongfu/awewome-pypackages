import flask


def process(data):
    return 'hello world!'

app = flask.Flask(__name__)
@app.route("/api/test", methods=["POST"])
def main():
    data = flask.request.form
    # 请求数据,主函数
    response_data = process(data)
    return response_data

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='6060',threaded=True)









