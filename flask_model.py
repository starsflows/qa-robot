import numpy as np
from flask import Flask, request, jsonify
import json
from main import chat

app=Flask(__name__)

@app.route('/func', methods=['POST', 'GET'])
def chatbot():
    chat(use_QA_first=False)

if __name__ =='__main__':
    app.run(host='127.0.0.1',port=8080)

