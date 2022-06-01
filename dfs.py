import numpy as np


def prior_prob(p1):
    return sum(p1) / len(p1)


def joint_prob(p1, p2):
    # return sum(np.multiply(p1,p2)) / len(p2)
    if all(p1 == p2):
        return prior_prob(p1)
    else:
        return prior_prob(p1 * p2)


def conditional_prob(p1, p2):
    """ conditional_prob(a, b) = P(A|B) """
    return joint_prob(p1, p2) / prior_prob(p2)


def inference_score(p1, p2):
    """
				 | (Pr(P|Q) - Pr(P)) / (1 - Pr(P))    iff Pr(P|Q) > Pr(P)
	  inf(P,Q) = |
				 | (Pr(P|Q) - Pr(P)) / Pr(P)          otherwise
	"""
    pr_ab = conditional_prob(p1, p2)
    pr_a = prior_prob(p1)
    if (pr_ab > pr_a):
        return (pr_ab - pr_a) / (1.0 - pr_a)
    else:
        return (pr_ab - pr_a) / pr_a


def negation(vector):
    return 1 - vector


def conjunction(vec1, vec2):
    return np.multiply(vec1, vec2)


def xor(vec1, vec2):
    return np.logical_xor(vec1, vec2).astype(int)


def disjunction(vec1, vec2):
    return np.logical_or(vec1, vec2).astype(int)  # numpy is faster


def implication(vec1, vec2):
    return disjunction(negation(vec1), vec2)


def equivalence(vec1, vec2):
    return np.equal(vec1, vec2)


def entails(vec1, vec2):
    """ Returns True if the the vector implication of a --> b contains only True (ie. entailment) and False otherwise
	"""
    return np.all(implication(vec1, vec2))
