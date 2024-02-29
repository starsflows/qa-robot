# -*- coding: UTF-8 -*-
import json
import sys
from sentence_transformers.util import cos_sim
import numpy as np
from sentence_transformers import SentenceTransformer as SBert

import sqlite3
import jieba
import logging

jieba.setLogLevel(logging.INFO)  # 设置不输出信息

conn = sqlite3.connect('./QA_data/QA.db')

cursor = conn.cursor()
stop_words = []
with open('./QA_data/stop_words.txt', encoding='gbk') as f:
    for line in f.readlines():
        stop_words.append(line.strip('\n'))


def match(input_question):
    res = []
    cnt = {}
    question = list(jieba.cut(input_question, cut_all=False))  # 对查询字符串进行分词
    for word in reversed(question):  # 去除停用词
        if word in stop_words:
            question.remove(word)
    for tag in question:  # 按照每个tag，循环构造查询语句
        keyword = "'%" + tag + "%'"
        result = cursor.execute("select * from QA where tag like " + keyword)
        for row in result:
            if row[0] not in cnt.keys():
                cnt[row[0]] = 0
            cnt[row[0]] += 1  # 统计记录出现的次数
    try:
        res_id = sorted(cnt.items(), key=lambda d: d[1], reverse=True)[0][0]  # 返回出现次数最高的记录的id
    except:
        return tuple()  # 若查询不出则返回空
    cursor.execute("select * from QA where id= " + str(res_id))
    res = cursor.fetchone()
    if type(res) == type(tuple()):
        return res  # 返回元组类型(id, question, answer, tag)
    else:
        return tuple()  # 若查询不出则返回空


def search_answer(input_question):
    model = SBert("F:\project\paraphrase-multilingual-MiniLM-L12-v2")
    with open('medical_data.json', 'r', encoding='utf-8') as f:
        database = json.load(f)
    question_embedding = model.encode(input_question)
    database_embedding = np.load('medical_data.npy')
    question_norm = np.linalg.norm(question_embedding)
    database_norm = np.linalg.norm(database_embedding, axis=1)
    # 直接利用向量计算相似度 question: [1,384] database: [n,384]
    sim = np.dot(question_embedding, database_embedding.T) / np.dot(database_norm, question_norm.T)
    # 返回最高相似度对应的id
    ids = np.argsort(-sim, axis=0)
    answer = database[int(ids[0])]['answer']
    if sim[ids].any() > 0.5:
        return answer
    else:
        return tuple()
