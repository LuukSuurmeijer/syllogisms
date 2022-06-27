import matplotlib.pyplot

from preprocessing import *
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import ccobra.syllogistic as ccobra
import sys
import matplotlib.pyplot as plt
import json

def matrix_to_dict(mat, idx_to_type, idx_to_response):
    ddict = defaultdict( lambda : defaultdict(lambda : 0))
    rowlen, columnlen = mat.shape
    for i in range(rowlen):
        for j in range(columnlen):
            type = idx_to_type[i]
            response = idx_to_response[j]
            ddict[type][response] = mat[i][j]
    return ddict

def find_syllogism(x):
    premise1, premise2 = [p.split(';') for p in x['task'].split('/')]
    response = x['response'].split(';')
    choices = [ccobra.encode_response(c.split(';'), (premise1, premise2)) for c in x['choices'].split('/')]

    task = ccobra.encode_task((premise1, premise2))
    conclusion = ccobra.encode_response(response, (premise1, premise2))

    correct = conclusion in choices

    return task, conclusion, correct

data = pd.read_csv('ccobra-master/benchmarks/syllogistic/data/Ragni2016.csv')
syllogism_simple = data.apply(find_syllogism, axis='columns', result_type='expand')
syllogism_simple.rename(columns={0:'task',1:'response_conclusion',2:'correct'})

# Some helper data structures to make working with numpy easy
type_to_idx = {syll : idx for idx, syll in enumerate(ccobra.SYLLOGISMS)}
response_to_idx = {syll : idx for idx, syll in enumerate(ccobra.RESPONSES)}
idx_to_type = ccobra.SYLLOGISMS
idx_to_response = ccobra.RESPONSES


correctness_dict = defaultdict(lambda : defaultdict(lambda : 0))
for i in syllogism_simple.iterrows():
    type, response, correct = i[1].array
    correctness_dict[type][response] += 1

# generate count matrix
response_count_mat = np.zeros((len(ccobra.SYLLOGISMS), len(ccobra.RESPONSES)))
for i in syllogism_simple.iterrows():
    type, response, correct = i[1].array
    row = type_to_idx[type]
    column = response_to_idx[response]
    response_count_mat[row][column] += 1

# now the proportional matrix
response_prop_mat = (response_count_mat.T / response_count_mat.sum(axis=1)).T
assert all(response_prop_mat.sum(axis=1))
new = matrix_to_dict(response_prop_mat, idx_to_type, idx_to_response)


response_indices = []
for type, responses in ccobra.SYLLOGISTIC_FOL_RESPONSES.items():
    response_ids = [response_to_idx[resp] for resp in responses]
    response_indices.append(response_ids)

THRESHOLD = 0.5

syll_categories = {'known_valid' : [], 'unknown_valid' : [], 'known_invalid' : [], 'unknown_invalid' : []}
for i in range(response_prop_mat.shape[0]):
    type = idx_to_type[i]
    NVC_idx = response_to_idx['NVC']
    prop_valid = sum(response_prop_mat[i][response_indices[i]])
    if prop_valid > THRESHOLD and type in ccobra.VALID_SYLLOGISMS:
        syll_categories['known_valid'].append((type, prop_valid))
    elif prop_valid < THRESHOLD and type in ccobra.VALID_SYLLOGISMS:
        syll_categories['unknown_valid'].append((type, prop_valid))
    elif response_prop_mat[i][NVC_idx] > THRESHOLD and not type in ccobra.VALID_SYLLOGISMS:
        syll_categories['known_invalid'].append((type, prop_valid))
    elif response_prop_mat[i][NVC_idx] < THRESHOLD and not type in ccobra.VALID_SYLLOGISMS:
        syll_categories['unknown_invalid'].append((type, response_prop_mat[i][NVC_idx]))

syll_colors = {'known_valid' : 'maroon', 'unknown_valid' : 'blue', 'known_invalid' : 'purple', 'unknown_invalid' : 'green'}
labels = list(syll_categories.keys())
handles = [plt.Rectangle((0,0),1,1, color=syll_colors[label]) for label in labels]
fig, axs = matplotlib.pyplot.subplots(2,2, sharey=True, figsize=(12,10))
axs[0,0].bar([key for key, val in syll_categories['known_valid']], [val for key, val in syll_categories['known_valid']], color=syll_colors['known_valid'])
axs[0,0].axhline(y=THRESHOLD, color='black', linestyle='--', lw=1, alpha=0.5)
axs[0,0].tick_params('x', labelrotation=25)
axs[0,0].set_xticklabels([key for key, val in syll_categories['known_valid']], horizontalalignment='right', fontsize=6)

axs[0,1].bar([key for key, val in syll_categories['known_invalid']], [val for key, val in syll_categories['known_invalid']], color=syll_colors['known_invalid'])
axs[0,1].axhline(y=THRESHOLD, color='black', linestyle='--', lw=1, alpha=0.5)

axs[1,0].bar([key for key, val in syll_categories['unknown_valid']], [val for key, val in syll_categories['unknown_valid']], color=syll_colors['unknown_valid'])
axs[1,0].axhline(y=THRESHOLD, color='black', linestyle='--', lw=1, alpha=0.5)
axs[1,0].tick_params('x', labelrotation=25)
axs[1,0].set_xticklabels([key for key, val in syll_categories['unknown_valid']], horizontalalignment='right', fontsize=6)


axs[1,1].bar([key for key, val in syll_categories['unknown_invalid']], [val for key, val in syll_categories['unknown_invalid']], color=syll_colors['unknown_invalid'])
axs[1,1].axhline(y=THRESHOLD, color='gray', linestyle='--', lw=1, alpha=0.5)
axs[1,1].tick_params('x', labelrotation=45)
axs[1,1].set_xticklabels([key for key, val in syll_categories['unknown_invalid']], horizontalalignment='right', fontsize=4)

# add legends and set its box position
fig.legend(handles, labels,
           bbox_to_anchor=(0.03, 0.5), loc='center left')
plt.subplots_adjust(left=0.27, right=0.9, wspace=0.15, hspace=0.25)

plt.savefig('plots/syllogism_categorization.pdf')

# the json file where the output must be stored
with open("syll_data/syll_categories.json", "w") as f:
    json.dump(syll_categories, f)


