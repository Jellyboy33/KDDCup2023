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
parser.add_argument('--locale', default='UK', help='Locale model=UK/DE/JP')
opt = parser.parse_args()
print(opt)

tra_dataset = 'KDDCup/sessions_train.csv'
tes_dataset = 'KDDCup/sessions_test_task1.csv'

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
tra_sessions = []
test_sessions = []

#print("-- Starting @ %ss" % datetime.datetime.now())
with open(tra_dataset, "r") as f:
    if opt.dataset == 'KDDCup':
        reader = csv.DictReader(f, delimiter=',')
    item_ctr = 1

    for data in reader:
        if data['locale']==opt.locale:
            i = data['prev_items'].split()
            j = data['next_item'].split()
            tempItems = i
            #print(tempItems)
            outseq = []
        
            for x in tempItems:
                x = ''.join(filter(str.isalnum, x))
                #print(x)
                if x in item_dict:
                    outseq += [item_dict[x]]
                else:
                    outseq += [item_ctr]
                    item_dict[x] = item_ctr
                    item_ctr += 1
            #print(outseq)
            j = ''.join(filter(str.isalnum, j))
            if j in item_dict:
                    outseq += [item_dict[j]]
            else:
                outseq += [item_ctr]
                item_dict[j] = item_ctr
                item_ctr += 1
            if not outseq:
                continue
            else:
                tra_sessions += [outseq]


with open(tes_dataset, "r") as f:
    if opt.dataset == 'KDDCup':
        reader = csv.DictReader(f, delimiter=',')

    for data in reader:
        if data['locale']==opt.locale:
            i = data['prev_items'].split()
            #j = data['next_item'].split()
            tempItems = i
            #print(tempItems)
            outseq = []
        
            for x in tempItems:
                x = ''.join(filter(str.isalnum, x))
                #print(x)
                if x in item_dict:
                    outseq += [item_dict[x]]
                else:
                    outseq += [item_ctr]
                    item_dict[x] = item_ctr
                    item_ctr += 1
            #print(outseq)
            if not outseq:
                continue
            else:
                test_sessions += [outseq]

print("Num_Nodes:" + str(item_ctr))
iid_counts = {}
for s in tra_sessions:
    for iid in s:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

for s in test_sessions:
    for iid in s:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(tra_sessions)
tempTrain = []
tempTest = []
for s in list(tra_sessions):
    filseq = list(filter(lambda i: iid_counts[i] >= 10, s))
    if len(filseq) < 2:
        continue
    else:
        s = filseq
        tempTrain += [s]

for s in list(test_sessions):
    filseq = list(filter(lambda i: iid_counts[i] >= 10, s))
    if len(filseq) < 2:
        continue
    else:
        s = filseq
        tempTest += [s]



test_sessions = tempTest
tra_sessions = tempTrain
print(len(iid_counts))
for key in list(iid_counts):
    if iid_counts[key] < 10:
        del iid_counts[key]
                
print (len(iid_counts))
print("-- Reading data @ %ss" % datetime.datetime.now())

def process_seqs_tra(iseqs):
    out_seqs = []
    labs = []
    for seq in iseqs:
        for i in range(1, len(seq)):
            tar = seq[-i]
        #    tar = seq[]
            labs += [tar]
            out_seqs += [seq[:-i]]
    return out_seqs, labs

def process_seqs_test(iseqs):
    out_seqs = []
    labs = []
    for seq in iseqs:
        #print(seq)
        tar=seq[-1]
        labs += [tar]
        out_seqs += [seq[:-1]]
    return out_seqs, labs


tra, target =  process_seqs_tra(tra_sessions)
tes, test_target =  process_seqs_test(test_sessions)

print(tra[:3],target[:3])
print(tes[:3],test_target[:3])


tra_final = (tra,target)
tes_final = (tes,test_target)

if opt.dataset == 'KDDCup':
    if not os.path.exists('KDDCup'):
        os.makedirs('KDDCup')
        pickle.dump(tra_final,open('KDDCup/train.txt','wb'))
        pickle.dump(tes_final,open('KDDCup/test.txt','wb'))
        #pickle.dump(tra,open('KDDCup/train.txt','wb'))
    else:
        pickle.dump(tra_final,open('KDDCup/train.txt','wb'))
        pickle.dump(tes_final,open('KDDCup/test.txt','wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    #pickle.dump(tes, open('sample/test.txt', 'wb'))
    #pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')


