"""
Functions used to generate the new tagging schema sequence
"""
import pandas as pd
import numpy as np

"""
Convert a list to a string
"""
def list_to_string(in_list):
    strg = ''
    strg = ' '.join([str(elem) for elem in in_list])
    return strg

"""
Compare whether the two sequences are same
"""
def compare_seq(seq_1, seq_2):
    if len(seq_1) != len(seq_2):
        return False
    else:
        for i in range(len(seq_1)):
            if seq_1[i] != seq_2[i]:
                return False
        return True

"""
Initiate the tagging sequence with only "O"
"""
def gen_tagging_seq(sent):
    init_tags = []
    l = len(sent.split(' '))
    for i in range(l):
        init_tags.append('O')
    res = list_to_string(init_tags)
    return res

"""
Initiate the dictionary to store each unique sentence
"""
def gen_init_dict(sents):
    dic = {}
    for sent in sents:
        if sent not in dic:
            dic[sent] = gen_tagging_seq(sent)
    return dic

"""
Generate tuples of (sentence, tag sequence) from datasets
"""
def gen_data_tuples(sents, tags):
    ds, dt = [], []
    for i in range(len(sents)):
        if len(sents[i]) > 0:
            ds.append(sents[i])
            dt.append(tags[i])
        else:
            continue
    dtuples = [(ds[i].split(), dt[i].split()) for i in range(len(ds))]
    return dtuples

"""
Add predicates into the tagging sequence
"""
def upd_tagging_seq(dic, dtuples):
    for i in range(len(dtuples)):
        sent = list_to_string(dtuples[i][0])
        dr = dtuples[i][1]
        dt = dic[sent].split(' ')
        for j in range(len(dr)):
            if dt[j] == 'O' and (dr[j] == 'P-B' or dr[j] == 'P-I'):
                dt[j] = dr[j]
        dic[sent] = list_to_string(dt)
    return dic

"""
Add arguments into the tagging sequence
"""
def upd_tagging_seq_arg(dic, dtuples):
    for i in range(len(dtuples)):
        sent = list_to_string(dtuples[i][0])
        dr = dtuples[i][1]
        dt = dic[sent].split(' ')
        for j in range(len(dr)):
            if dt[j] == 'O' and (dr[j][0] == 'A' and dr[j][-1] == 'B'):
                dt[j] = 'A-B'
            if dt[j] == 'O' and (dr[j][0] == 'A' and dr[j][-1] == 'I'):
                dt[j] = 'A-I'
        dic[sent] = list_to_string(dt)
    return dic

"""
Make predicates in order P0-B, P1-B, P2-B...
"""
def add_order_to_pred(seq):
    arr = seq.split()
    ord_pb = 0
    ord_pi = 0
    for i in range(len(arr)):
        if arr[i] == 'P-B':
            arr[i] = 'P' + str(ord_pb) + '-B'
            ord_pi = ord_pb
            ord_pb += 1
        elif arr[i] == 'P-I':
            arr[i] = 'P' + str(ord_pi) + '-I'
            
    return arr

"""
Make arguments in order A0-B A0-I A1-B A1-I A2-B A2-I...
"""
def add_order_to_args(seq):
    arr = seq.split()
    ord_pb = 0
    ord_pi = 0
    for i in range(len(arr)):
        if arr[i] == 'A-B':
            arr[i] = 'A' + str(ord_pb) + '-B'
            ord_pi = ord_pb
            ord_pb += 1
        elif arr[i] == 'A-I':
            arr[i] = 'A' + str(ord_pi) + '-I'
            
    return arr

"""
Generate the new tagging sequence with multiple predicates (relations)
"""
def gen_tagging_res(sents, tags):
    dtuples = gen_data_tuples(sents, tags)
    dic = gen_init_dict(sents)
    dic = upd_tagging_seq(dic, dtuples)
    data_s, data_t = [], []
    for k in dic:
        if len(k) > 0:
            data_s.append(k)
            data_t.append(dic[k])
    data = [(data_s[i].split(), add_order_to_pred(data_t[i])) for i in range(len(data_s))]
    return data