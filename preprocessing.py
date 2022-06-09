import matplotlib.pyplot as plt
import matplotlib as mpl
import ccobra
from itertools import combinations, product
import pandas as pd
import numpy as np
import dfs
import re
import json
import matplotlib.pyplot as plt
import ccobra
import pandas as pd
import dfs
from functools import reduce
from matplotlib_venn import venn3, venn3_circles
import itertools as it
import matplotlib as mpl
import descartes
import shapely.geometry as sg
import shapely.ops as so
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


def getConclusionSem(p1, p2, conclusion, size, dict):
    resp = ccobra.syllogistic.decode_response(conclusion, (p1, p2))[0]
    if resp[0] == 'NVC':
        return resp, np.zeros((size))
    else:
        conclusion_sem = dict[' '.join(resp)]
        return resp, np.array(conclusion_sem, dtype=int)

def find_inference(x, response, size, dict):
    score = 0
    conclusion_sem = getConclusionSem(x['premise1'], x['premise2'], response, size, dict)
    if conclusion_sem[0][0] == 'NVC':
        #score = None
        #for r in ccobra.syllogistic.RESPONSES[:-1]:
        #    resp = ccobra.syllogistic.decode_response(r, (['All', 'trumpeters', 'pianists'], ['All', 'trumpeters', 'singers']))
        #    print(resp, r, x['premise1'], x['premise2'])
        #    conclusion_sem = sen_sem_dict[' '.join(resp[0])]
        #    if dfs.inference_score(conclusion_sem, dfs.conjunction(x['sem1'], x['sem2'])) == 1:
        #        score = -1
        #        break
        #if not score:
        #    score = 1
        if x['type'] in ccobra.syllogistic.VALID_SYLLOGISMS:
            score = -1
        else:
            score = 1

    else:
        score = dfs.inference_score(conclusion_sem[1], dfs.conjunction(np.array(x['sem1']), np.array(x['sem2'])))
    return score

def makeDiscourse(p1, p2, c): #stupid call by reference
    discourse = ' <EOS> '.join([' '.join(p1), ' '.join(p2), ' '.join(c)])
    return ''.join((discourse, ' <EOS>'))

def generate_data(name, write=True):
    sent_raw = read_mesh(name)
    # remove all premises of the form '[Q] [A] are [A]'
    sent_notaut = [tup for tup in sent_raw if tup[0][-1] != tup[0][-2]]
    # get all combinations to form doubles
    pairs = [double for double in list(product(sent_notaut, sent_notaut))]
    # remove all doubles that do not contain distinct A, B and C
    pairs_unique = [double for double in pairs if len(set([*double[0][0][1:], *double[1][0][1:]])) == 3]
    print(pairs_unique[0])
    sen_sem_dict = {' '.join(tup[0]) : tup[1] for tup in sent_notaut}
    correct_responses = ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES

    # Here we generate the syllogism premises and corresponding semantics
    if write:
        full = []
        for index,double in enumerate(pairs_unique):
            task = ccobra.syllogistic.encode_task((double[0][0], double[1][0]))
            full.append((index, task, double[0][0], double[1][0], double[0][1], double[1][1]))
        data = pd.DataFrame(full, columns=['index', 'type','premise1','premise2', 'sem1', 'sem2'])
        data.to_json("premise_info.json")
    return sen_sem_dict

def inference_matrix(data, types, correct_responses):
    fig = plt.figure(figsize=(1,1), dpi=180)
    ax = fig.add_subplot(111)
    mat = ax.matshow(data.T, cmap=plt.get_cmap('RdBu'), aspect='auto')
    cb = fig.colorbar(mat, location='top', shrink=0.35, ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=32)

    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_xticklabels(types, rotation=0, fontsize=30)
    ax.set_yticklabels(data.columns.tolist(), rotation=0, fontsize=36)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)

    fig.align_xlabels()

    x_labels = list(ax.get_xticklabels())
    y_labels = list(ax.get_yticklabels())
    min_val, max_val, diff = 0., data.shape[0], 1.
    ind_array = np.arange(min_val, max_val, diff)
    ind_array_y = np.arange(min_val, data.shape[1], diff)
    x, y = np.meshgrid(ind_array, ind_array_y)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if y_labels[int(y_val)].get_text() in correct_responses[x_labels[int(x_val)].get_text()]:
            plt.text(x_val, y_val, '*', va='center', ha='center', color='gray', fontsize=42)


    fig.set_size_inches(80, 20, forward=True)

    plt.savefig('test.png')
    plt.show()

# for all the Venn stuff, I follow this tutorial http://drsfenner.org/blog/2015/04/visualizing-probabilities-in-a-venn-diagram-2/
def makeShapelyCircles(numEvents, size=2.0):
    thetas = np.linspace(0, int(2*np.pi - (2*np.pi/int(numEvents))), int(numEvents))
    r = np.ones(thetas.shape)
    centers = np.column_stack([r*np.cos(thetas), r*np.sin(thetas)])
    return [sg.Point(*c).buffer(float(size)) for c in centers]

# patch = [p1, p2, p3, ...]
# color = [c1, c2, c3, ...]
def addPatchesAndColors(ax, patch, color):
    addPatchColorPairs(ax, it.izip(patch, color))

# pAndC = [(patch, color), (patch, color)]
def addPatchColorPairs(ax, pAndC):
    cmap = plt.cm.get_cmap('RdBu')
    for patch, color in pAndC:
        color = mpl.colors.to_hex(cmap(np.array([color], dtype=float)))
        ax.add_patch(descartes.PolygonPatch(patch,
                                            fc=color,
                                            ec='black'))

# for bonus points, you could try to replace this with
# a clever reduce/functools.reduce call
def chainIntersection(comps):
    #comps = iter(comps)
    #intersection = comps.next()
    #for comp in comps:
    #    intersection = comp.intersection(intersection)
    return reduce(lambda p,q : p.intersection(q), comps)

# more examples of the indexing scheme:
# index ---> event indicator ---> which events occurred
# 0 --> 0,0,0,0  []

# 1 --> 1,0,0,0  [0]
# 2 --> 0,1,0,0  [1]
# 3 --> 1,1,0,0  [0,1]
# 4 --> 0,0,1,0  [2]

# 5 --> 1,0,1,0  [0,2]

# 15  --> 1,1,1,1  [0,1,2,3]
def makePatchColors(events, probs):
    eventCt  = len(events)
    pwrSetCt = 2**eventCt
    primEvents = (np.indices([2] * eventCt, dtype=np.uint8)
                    .reshape(eventCt, pwrSetCt)[::-1].T)

    GC = sg.collection.GeometryCollection
    res = []
    for anEvent, prob in zip(primEvents, probs):
        # UGLY!
        indices = [idx for idx, val in enumerate(anEvent) if val]
        if not indices:
            continue
        components = GC([events[idx] for idx in indices])
        patch = chainIntersection(components)
        res.append((patch, prob)) # color/prob inverted
    return res

def makeRandomProbs(numEvents, bigNull = False):
    probs = np.random.uniform(0.0, 10.0, 2**numEvents)
    if bigNull: # hack null event to big prob.
        probs[0] = .5 * probs[1:].sum()
    probs = probs / probs.sum()
    print("sorted probs: %s sum: %d" % (np.array(sorted(probs)), probs.sum()))

    return probs

def addColorbar(ax, cmapName):
    cbax, cbkw = mpl.colorbar.make_axes(ax)
    mpl.colorbar.ColorbarBase(cbax,
                              cmap=plt.cm.get_cmap(cmapName),
                              norm=mpl.colors.Normalize(clip=True,
                                                        vmin=-1.0,
                                                        vmax=1.0),
                              **cbkw)

def addVennDiagram(ax, probs):
    cmap = plt.cm.get_cmap('RdBu')
    circs = makeShapelyCircles(np.log2(len(probs)))
    pacs   = makePatchColors(circs, probs)
    addPatchColorPairs(ax, pacs)

    ax.set_facecolor(color = mpl.colors.to_hex(cmap(np.array([probs[0]], dtype=float))))


    # fig.colorbar(mplPatches)
    addColorbar(ax, "RdBu")

def createWith(probs):
    #
    # control display
    #
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    addVennDiagram(ax, probs)

    ax.plot()
    plt.savefig('venn_test.pdf')
    plt.show()