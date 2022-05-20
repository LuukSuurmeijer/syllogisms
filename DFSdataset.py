import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import combinations, product, chain
from functools import reduce
import dfs
import ccobra
import os
import re
import csv
import sys

# TODO: Some not /Some_not etc.

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

class Syllogism:
    def __init__(self, p1, p2, c):
        self.premises = [p1, p2]
        self.conclusion = c
        self.full_form = [p1, p2, c]
        self.task = ccobra.syllogistic.encode_task(self.premises)
        self.conclusion_type = ccobra.syllogistic.encode_response(c, self.premises)

    def is_valid(self):
        if self.conclusion in ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES:
            return True
        else:
            False

    def __repr__(self):
        return str(self.full_form)

class DFSDataset(Dataset):
    """Class that takes a mesh file and produces a pytorch dataset."""
    def __init__(self, mesh_file, delim):
        self.pairs, self.sen_sem_dict = self.generate_semantics(read_mesh(mesh_file))
        self.data = self.generate_IO_pairs()
        self.delim = delim
        self.word2id, self.id2word = self.genVocab(self.data)

    def makeDiscourse(self, p1, p2, c): #stupid call by reference
        discourse = f' {self.delim} '.join([' '.join(p1), ' '.join(p2), ' '.join(c)])
        return ''.join((discourse, f' {self.delim}'))

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

    def getConclusionSem(self, syllogism):
        """
        Returns the semantics for a given conclusion. Need the premises to find the right conclusion template.
        """
        resp = syllogism.conclusion
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
                response = ccobra.syllogistic.decode_response(response, (p1[0], p2[0]))[0]
                syllogism = Syllogism(p1[0], p2[0], response)
                conclusion = self.getConclusionSem(syllogism)
                target_semantics = reduce(dfs.conjunction, [p1[1], p2[1], conclusion[1]])
                all_io_pairs.append((syllogism, target_semantics))
        return all_io_pairs

    def genVocab(self, raw):
        """
        Generates a list that maps indices onto words, and a dictionary that maps words onto indices
        """
        word2id = {}
        id2word = []
        vocab = set(' '.join([word for syll, sem in raw for word in list(chain(*syll.full_form))]).split()) #wtf am i even doing here
        print(vocab)
        for i, word in enumerate(vocab):
            word2id[word] = i
            id2word.append(word)
        id2word.append(self.delim)
        word2id[self.delim] = len(vocab)
        return (word2id, id2word)

    def make_one_hot(self, sentence):
        """
        Generates a one-hot encoded vector from a list of strings. Strings must be in self.vocab
        """
        one_hot = torch.zeros(len(sentence), len(self.word2id))
        for i, word in enumerate(sentence):
            one_hot[i][self.word2id[word]] = 1
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

    def decode_training_item(self, item):
        idx = [i for i,word in enumerate(item) if word == self.delim]
        sentences = [item[s+1:e] for s,e in zip([-1] + idx, idx + [len(item)])][:-1]
        return Syllogism(*sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Yields the example at index `idx`. An example is a 2-tuple.
        First element is a one-hot-encoded input sentence of [len, vocabsize]
        Second element is a DFS vector of [len, model_size]
        """
        syllogism, semantics = self.data[idx]
        full_discourse = self.makeDiscourse(*syllogism.full_form).split()
        return self.make_one_hot(full_discourse), semantics.repeat(len(full_discourse),1)


test = DFSDataset('dfs_data/syllogism_10k.mesh', '<EOS>')
item1 = test[0]
#test.decode_training_item(item1)

for idx, ex in enumerate(test):
    print(test.decode_training_item(test.translate_one_hot(ex[0])), dfs.prob(ex[1][0]))
