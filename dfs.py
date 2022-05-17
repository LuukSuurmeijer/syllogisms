import torch
import numpy as np
import pandas as pd

def prob(p1):
    return sum(p1)/len(p1)

def jointProb(p1, p2):
    return sum(np.multiply(p1,p2)) / len(p2)

def conditionalProb(p1, p2):
    """ conditional_prob(a, b) = P(A|B) """
    return jointProb(p1, p2) / prob(p2)

def inferenceScore(p1, p2):
    """
                 | (Pr(P|Q) - Pr(P)) / (1 - Pr(P))    iff Pr(P|Q) > Pr(P)
      inf(P,Q) = |
                 | (Pr(P|Q) - Pr(P)) / Pr(P)          otherwise
    """
    posterior = conditionalProb(p1, p2)
    prior = prob(p1)
    if posterior > prior:
        return (posterior - prior) / (1 - prior)
    else:
        return (posterior - prior) / prior

def negation(vector):
    return 1-vector

def conjunction(vec1, vec2):
    return np.multiply(vec1, vec2)

def xor(vec1, vec2):
		return np.logical_xor(vec1,vec2).astype(int)

def disjunction(vec1, vec2):
    return np.logical_or(vec1, vec2).astype(int) #numpy is faster

def implication(vec1, vec2):
    return disjunction(negation(vec1), vec2)

def equivalence(vec1, vec2):
    return np.equal(vec1, vec2)

def entails(vec1, vec2):
    """ Returns True if the the vector implication of a --> b contains only True (ie. entailment) and False otherwise
    """
    return np.all(implication(vec1, vec2))

### Meaning Space object ###
class MeaningSpace:
    def __init__(self, file):
        self.matrix, self.idx2prop, self.prop2idx = self.readSpace(file) #matrix, list of propositions (strings), dict of indices to propositions (strings)
        self.shape = self.matrix.shape

    def __len__(self):
        return len(self.propositions)

    def readSpace(self, file):
        """
        Read a space from a .observations file
        """
        model_df = pd.read_csv(file, sep=' ', header=0)
        model_matrix = np.array(model_df)
        names = list(model_df.columns)
        idx2prop = names
        prop2idx = {prop : idx for idx, prop in enumerate(names)}
        return model_matrix, idx2prop, prop2idx

    def getVector(self, propname):
        idx = self.prop2idx[propname]
        return self.matrix[:,idx]
