import torch
from torch.utils.data import Dataset
from itertools import product, chain
from functools import reduce
import dfs
import ccobra
import re
import sys

# TODO: Some not /Some_not in Syllogism class clashes


def read_mesh(file):
    out = []
    with open(f"{file}", 'r') as f:
        contents = f.read().split('\n')
        for i in range(len(contents)):
            if contents[i].startswith('Name'):
                match = re.search(r'\".+\"', contents[i])
                ex_name = match[0].strip('\"').split()
                ex_name[0] = ex_name[0].replace('_', ' ')
                for j in range(i, len(contents)):
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
        if p2 != [] and c != []:
            self.task = ccobra.syllogistic.encode_task(self.premises)
            self.conclusion_type = ccobra.syllogistic.encode_response(c, self.premises)

    def is_valid(self):
        if self.conclusion_type in ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[self.task]:
            return True
        else:
            return False

    def __getitem__(self, idx):
        return self.full_form[idx]

    def __repr__(self):
        return str(self.full_form)


class Vocabulary:
    def __init__(self, data, delim, phrasal=False):
        self.phrasal = phrasal
        if self.phrasal:
            sentences = list(chain(*[sent for sent, sem in data]))
            self.v = set([' '.join(sent) for sent in sentences])
        else:
            self.v = set(list(chain(*[' '.join(word).split() for sent, sem in data for word in sent])))
        self.delim = delim
        self.word2id, self.id2word = self.gen_vocab()

    def gen_vocab(self):
        """
        Generates a list that maps indices onto words, and a dictionary that maps words onto indices
        """
        word2id = {}
        id2word = []
        for i, word in enumerate(self.v):
            word2id[word] = i
            id2word.append(word)
        id2word.append(self.delim)
        word2id[self.delim] = len(self.v)
        return word2id, id2word

    def make_one_hot(self, sentence):
        """
        Generates a one-hot encoded vector from a list of strings. Strings must be in self.vocab
        """
        #if self.phrasal:
        #    length = 1
        #    sentence = [sentence[0]]
        #else:
        #    length = len(sentence)
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

    def __len__(self):
        return len(self.word2id)

class DFSdataset(Dataset):
    def __init__(self, mesh_data, delim):
        if isinstance(mesh_data, str):
            pairs, self.sen_sem_dict, self.obs_size = self.generate_semantics(read_mesh(mesh_data))
            self.data = self.generate_IO_pairs(pairs)
        else:
            # if data is not read from a meshfile, it is assumed to be given as a tuple of
            # ( sent_sem_pairs, sent_sem_dict)
            # TODO: How to represent single premises that do not constitute syllogisms
            data, self.sen_sem_dict = mesh_data
            self.data = [(tup[0], tup[1]) for tup in data]
            self.obs_size = len(list(self.sen_sem_dict.values())[0])
        self.delim = delim
        self.sen_sem_dict['NVC'] = torch.zeros(self.obs_size) + 1/self.obs_size

    @staticmethod
    def generate_semantics(sent_raw):
        """
        This function returns a list of 2-tuples where
        the first element is a legal premise pair and the second element is their (independent) semantics (vectors)
        As well as a dictionary that maps propositions onto semantic vectors
        """
        # remove all premises of the form '[Q] [A] are [A]'
        sent_notaut = [tup for tup in sent_raw if tup[0][-1] != tup[0][-2]]
        # get all combinations to form doubles
        pairs = [double for double in list(product(sent_notaut, sent_notaut))]
        # remove all doubles that do not contain distinct A, B and C
        pairs_unique = [double for double in pairs if len({*double[0][0][1:], *double[1][0][1:]}) == 3]
        sen_sem_dict = {' '.join(tup[0]): tup[1] for tup in sent_notaut}
        return pairs_unique, sen_sem_dict, len(pairs_unique[0][0][1])

    def get_conclusionsem(self, syllogism):
        """
        Returns the semantics for a given conclusion. Need the premises to find the right conclusion template.
        """
        resp = syllogism.conclusion
        if resp[0] == 'NVC':
            return resp, torch.zeros(self.obs_size) + 1/self.obs_size
        else:
            conclusion_sem = self.sen_sem_dict[' '.join(resp)]
            return resp, torch.DoubleTensor(conclusion_sem)

    def generate_IO_pairs(self, pairs):
        """
        Generates the data, that is, triples of 2 premises with any conclusion with the semantics corresponding
        to the conjunction of their independent semantic vectors.
        list of 2-tuples where first element is string of the full discourse and second element a semantic vector.
        My data: 384 premise pairs x 9 conclusions = 3456 unique examples

            pairs: list of premise-pairs and DFS-vectors
        """
        all_io_pairs = []
        for idx, pair in enumerate(pairs):  # this is slow af but its only preprocessing anyway
            p1, p2 = pair[0], pair[1]  # p1 is a tuple of (list, vector)
            for response in ccobra.syllogistic.RESPONSES:
                response = ccobra.syllogistic.decode_response(response, (p1[0], p2[0]))[0]
                syllogism = Syllogism(p1[0], p2[0], response)
                conclusion = self.get_conclusionsem(syllogism)
                # if the resultant conclusion is the zero vector (cause contradiction), just have it be an extremely
                # unlikely event
                target_semantics = reduce(dfs.conjunction, [p1[1], p2[1], conclusion[1]])
                if not target_semantics.any():
                    target_semantics = torch.zeros(self.obs_size) + 1/self.obs_size
                all_io_pairs.append((syllogism, target_semantics))
        return all_io_pairs

    def decode_training_item(self, item):
        """
            Turns a sequence of words/phrases into a Syllogism object. In this class cause of self.delim

            item: the triple of sentences to decode as a list of lists of strings
        """
        #TODO: What if delim = None or empty string? Then this does not work.
        idx = [i for i, word in enumerate(item) if word == self.delim]
        sentences = [item[s + 1:e] for s, e in zip([-1] + idx, idx + [len(item)])][:-1]
        # ccobra uses space for 'some not' quantifier, terrible encoding, so here is some terrible code to fix it
        for sent in sentences:
            if 'not' in sent:
                sent.remove('not')
                sent[0] = ' '.join([sent[0], 'not'])
        return sentences

    def __len__(self):
        return len(self.data)

class DFSdatasetWord(DFSdataset):
    """Class that takes a mesh file and produces a pytorch dataset."""

    def __init__(self, mesh_data, delim):
        super().__init__(mesh_data, delim)
        self.vocab = Vocabulary(self.data, self.delim, phrasal=False)

    def make_discourse(self, p1, p2, c):  # stupid call by reference
        discourse = f' {self.delim} '.join([' '.join(p1), ' '.join(p2), ' '.join(c)])
        return ''.join((discourse, f' {self.delim}'))

    def __getitem__(self, idx):
        """
        Yields the example at index `idx`. An example is a 2-tuple.
        First element is a one-hot-encoded input sentence of [len, vocabsize]
        Second element is a DFS vector of [len, model_size]

            idx: index of self.data to return
        """
        syllogism, semantics = self.data[idx]
        full_discourse = self.make_discourse(*syllogism.full_form).split()
        return self.vocab.make_one_hot(full_discourse).double(), semantics.repeat(len(full_discourse), 1).double()


class DFSdatasetPhrase(DFSdataset):
    def __init__(self, mesh_data, delim):
        super().__init__(mesh_data, delim)
        self.vocab = Vocabulary(self.data, '', phrasal=True)

    def make_discourse(self, p1, p2, c):
        discourse = [' '.join(p1), ' '.join(p2), ' '.join(c)]
        return discourse

    def __getitem__(self, idx):
        syllogism, semantics = self.data[idx]
        full_discourse = self.make_discourse(*syllogism.full_form)
        return self.vocab.make_one_hot(full_discourse).double(), semantics.repeat(len(full_discourse), 1).double()