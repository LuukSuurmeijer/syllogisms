import matplotlib.pyplot as plt
import matplotlib as mpl
import ccobra
from itertools import combinations, product
import pandas as pd
import numpy as np
import dfs
import re
import csv
from functools import reduce

def read_mesh(file):
        out = []
        with open(f"{file}", 'r') as f:
            contents = f.read().split('\n')
            for i in range(len(contents)):
                if contents[i].startswith('Name'):
                    match = re.search(r'\".+\"', contents[i])
                    ex_name = match[0].strip('\"').split()
                    ex_name[0] = ex_name[0].replace('_', ' ')
                    for j in range(i,len(contents)):
                        if contents[j].startswith('Input'):
                            match = re.search(r'Target ([0 1]+)', contents[j])
                            ex_target = [int(elem) for elem in match[0].strip('Target ').split()]
                            break
                    out.append((ex_name, np.array(ex_target)))
        return out

def getConclusionSem(p1, p2, conclusion):
    resp = ccobra.syllogistic.decode_response(conclusion, (p1, p2))[0]
    if resp[0] == 'NVC':
        return resp, np.zeros((10000))
    else:
        conclusion_sem = sen_sem_dict[' '.join(resp)]
        return resp, np.array(conclusion_sem)

def find_inference(x):
    try:
        conclusion_sem = getConclusionSem(x['premise1'], x['premise2'], response)
    except KeyError:
        conclusion_sem = 'NVC'

    if conclusion_sem == 'NVC':
        score = None
        for r in ccobra.syllogistic.RESPONSES[:-1]:
            resp = ccobra.syllogistic.decode_response(r, (x['premise1'], x['premise2']))
            conclusion_sem = sen_sem_dict[' '.join(resp[0])]
            if dfs.inferenceScore(conclusion_sem, dfs.conjunction(x['sem1'],x['sem2'])) == 1:
                score = -1
                break
        if not score:
            score = 1

    else:
        score = dfs.inferenceScore(conclusion_sem, dfs.conjunction(x['sem1'],x['sem2']))

    return score

def makeDiscourse(p1, p2, c): #stupid call by reference
    discourse = ' <EOS> '.join([' '.join(p1), ' '.join(p2), ' '.join(c)])
    return ''.join((discourse, ' <EOS>'))

sent_raw = read_mesh('dfs_data/syllogism_10k.mesh')
# remove all premises of the form '[Q] [A] are [A]'
print(sent_raw[0])
sent_notaut = [tup for tup in sent_raw if tup[0][-1] != tup[0][-2]]
print(sent_notaut[0])
# get all combinations to form doubles
pairs = [double for double in list(product(sent_notaut, sent_notaut))]
# remove all doubles that do not contain distinct A, B and C
pairs_unique = [double for double in pairs if len(set([*double[0][0][1:], *double[1][0][1:]])) == 3]
print(pairs_unique[0])
sen_sem_dict = {' '.join(tup[0]) : tup[1] for tup in sent_notaut}
correct_responses = ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES

# Here we generate the syllogism premises and corresponding semantics
full = []
for index,double in enumerate(pairs_unique):
    task = ccobra.syllogistic.encode_task((double[0][0], double[1][0]))
    full.append((index, task, double[0][0], double[1][0], double[0][1], double[1][1]))
data = pd.DataFrame(full, columns=['index', 'type','premise1','premise2', 'sem1', 'sem2'])
data.to_csv("premise_info.csv")
