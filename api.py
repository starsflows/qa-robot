import json

from flask import Flask, request, jsonify
from QA_data.QA_test import search_answer
from config import Config
import train_eval

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False


def chat(question, **kwargs):
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)

    searcher, sos, eos, unknown, word2ix, ix2word = train_eval.test(opt)

    input_sentence = question
    if opt.use_QA_first:
        query_res = search_answer(input_sentence)
        if (query_res == tuple()):
            output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix,
                                                    ix2word)
        else:
            # output_words = "您是不是要找以下问题: " + query_res[1] + '，您可以尝试这样: ' + query_res[2]
            output_words = query_res
    else:
        output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
    return output_words


@app.route('/', methods=['GET', 'POST'])
def return_result():
    # 接受数据
    data = request.get_data()
    data = json.loads(data)
    question = data['question']

    # 生成答案
    answer = chat(question, use_QA_first=True)

    # 返回答案
    return json.dumps({'answer': answer})


if __name__ == '__main':
    app.run()
