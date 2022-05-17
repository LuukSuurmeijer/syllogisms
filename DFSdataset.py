import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import combinations, product
from functools import reduce
import dfs
import ccobra
import os
import re
import csv
import sys

def makeDiscourse(p1, p2, c): #stupid call by reference
    discourse = ' <EOS> '.join([' '.join(p1), ' '.join(p2), ' '.join(c)])
    return ''.join((discourse, ' <EOS>'))

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
                    out.append((ex_name, torch.DoubleTensor(ex_target)))
        return out

class DFSDataset(Dataset):
    """Class that takes a mesh file and produces a pytorch dataset.
        repetition controls how many times each example is repeated in the dataset."""

    def __init__(self, mesh_file):
        self.pairs, self.sen_sem_dict = self.generate_semantics(read_mesh(mesh_file))
        self.data = self.generate_IO_pairs()
        self.word2id, self.id2word = self.genVocab(self.data)

    def generate_semantics(self, sent_raw):
        """
        This function returns a list of 2-tuples where the first element is a legal premise pair and the second element is their (independent) semantics (vectors)
        As well as a dictionary that maps propositions onto semantic vectors
        """
        # remove all premises of the form '[Q] [A] are [A]'
        sent_notaut = [tup for tup in sent_raw if tup[0][-1] != tup[0][-2]]
        # get all combinations to form doubles
        pairs = [double for double in list(product(sent_notaut, sent_notaut))]
        # remove all doubles that do not contain distinct A, B and C
        pairs_unique = [double for double in pairs if len(set([*double[0][0][1:], *double[1][0][1:]])) == 3]
        sen_sem_dict = {' '.join(tup[0]) : tup[1] for tup in sent_notaut}
        return pairs_unique, sen_sem_dict

    def getConclusionSem(self, p1, p2, conclusion):
        """
        Returns the semantics for a given conclusion. Need the premises to find the right conclusion template.
        """
        resp = ccobra.syllogistic.decode_response(conclusion, (p1, p2))[0]
        if resp[0] == 'NVC':
            return resp, torch.zeros((10000))
        else:
            conclusion_sem = self.sen_sem_dict[' '.join(resp)]
            return resp, torch.DoubleTensor(conclusion_sem)

    def generate_IO_pairs(self):
        """
        Generates the data, that is, triples of 2 premises with any conclusion with the semantics corresponding to the conjunction of their independent semantic vectors.
        list of 2-tuples where first element is string of the full discourse and second element a semantic vector.
        My data: 384 premise pairs x 9 conclusions = 3456 unique exampels
        """
        all_io_pairs = []
        for idx, pair in enumerate(self.pairs): #this is slow af but its only preprocessing anyway
            p1, p2 = pair[0], pair[1] #p1 is a tuple of (list, vector)
            for response in ccobra.syllogistic.RESPONSES:
                conclusion = self.getConclusionSem(p1[0], p2[0], response)
                target_semantics = reduce(dfs.conjunction, [p1[1], p2[1], conclusion[1]])
                target_discourse = makeDiscourse(p1[0], p2[0], conclusion[0])
                all_io_pairs.append((target_discourse, target_semantics))
        return all_io_pairs

    def genVocab(self, raw):
        """
        Generates a list that maps indices onto words, and a dictionary that maps words onto indices
        """
        word2id = {}
        id2word = []
        vocab = set(' '.join([name for name, label in raw]).split())
        for i, word in enumerate(vocab):
            word2id[word] = i
            id2word.append(word)
        return (word2id, id2word)

    def make_one_hot(self, sentence):
        """
        Generates a one-hot encoded vector from a list of strings. Strings must be in self.vocab
        """
        one_hot = torch.zeros(len(sentence), len(self.word2id))
        for i in range(len(sentence)):
            one_hot[i][self.word2id[sentence[i]]] = 1
        return one_hot

    def translate_one_hot(self, tensor):
        """
        Translates a one-hot encoded vector into words using self.vocab
        """
        sent = []
        for i in range(tensor.shape[0]):
            idx = torch.argmax(tensor[i])
            sent.append(self.id2word[idx])
        return sent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Yields the example at index `idx`. An example is a 2-tuple.
        First element is a one-hot-encoded input sentence of [len, vocabsize]
        Second element is a DFS vector of [len, model_size]
        """
        sentence, semantics = self.data[idx]
        split_sentence = sentence.split()
        return self.make_one_hot(split_sentence), semantics.repeat(len(split_sentence),1)


test = DFSDataset('dfs_data/syllogism_10k.mesh')
item1 = test[0]

for idx, ex in enumerate(test):
    print(test.translate_one_hot(ex[0]), dfs.prob(ex[1][0]))
