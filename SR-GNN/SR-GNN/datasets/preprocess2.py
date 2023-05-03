#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'KDDCup/sessions_train.csv'

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'KDDCup':
        reader = csv.DictReader(f, delimiter=',')
    sess_clicks = {}
    sessid = 0
    ctr = 0
    curid = -1
    item_ctr = 1

    sessions=[]
    for data in reader:     
        curid = sessid
        if opt.dataset == 'KDDCup':
             for i in data['prev_items']:
                outseq = []
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
                sessions += [outseq]
                

print("-- Reading data @ %ss" % datetime.datetime.now())



tra, target, id =  process_seqs(sessions)

if opt.dataset == 'KDDCup':
    if not os.pathexists('KDDCup'):
        os.makedirs('KDDCup')
        pickle.dump(tra,open('KDDCup/train.txt','wb'))
        pickle.dump(tes,open('KDDCup/test.txt','wb'))
        #pickle.dump(tra,open('KDDCup/train.txt','wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')


def process_seqs(iseqs):
    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs, ids