import pandas as pd
from .struct import *

from collections import namedtuple


# In[30]:


node_weights = defaultdict(lambda: 1)

def pops(l):
    if l:
        return l[0]
    return None

def collects(l):
    return [y for y in l if y]

def n_or_v2(synset):
    return synset.pos() in [wn.NOUN, wn.VERB]

def n_or_v(synset):
    return synset.pos() in [wn.NOUN]

def get_pos_men(x):
    x = x.lower()
    if x in "nv":
        return x
    elif x == "j":
        return "ars"
    else:
        return "nvars"

def hybrid_wup(s1, s2):
    if n_or_v2(s1.content) and n_or_v2(s2.content):
        return s1.content.wup_similarity(s2.content)
    return s1.wupalmer(s2)

def resnik(s1, s2, ic_corpus):
    if n_or_v(s1) and n_or_v(s2):
        return s1.res_similarity(s2, ic_corpus)
    return -1 #s1.wup_similarity(s2)

def _list_fallback(func, fallback_func, args, both=False):
    result = func(*args)
    if not result or both:
        result += fallback_func(*args)
    return result

sim_strategy = {
    "mfs": lambda x, p: [SemNode.make(pops(wn.synsets(x, p)))],
    "average": lambda x, p: [SemNode.make(s) for s in wn.synsets(x, p)],
    "word": lambda x, p: [SemNode.make(v) for v in tripsont().get_word(x, p)],
    "lookup": lambda x, p: _list_fallback(
        lambda x, p: [SemNode.make(v) for v in tripsont().get_word(x, p)],
        lambda x, p: [SemNode.make(pops(wn.synsets(x, p)))],
        [x, p]
    ),
    "lookupall": lambda x, p: _list_fallback(
        lambda x, p: [SemNode.make(v) for v in tripsont().get_word(x, p)],
        lambda x, p: [SemNode.make(pops(wn.synsets(x, p)))],
        [x, p], both=True
    ),
    "both": lambda x, p: [SemNode.make(v) for v in tripsont().get_word(x, p)] + [SemNode.make(pops(wn.synsets(x, p)))]
}

sim_metric = {
    "cross" : lambda x, y: x.wupalmer(y, node_weights),
    "tripspath" : lambda x, y: x.path_similarity(y, weights=node_weights),
    "normal": lambda x, y: x.content.wup_similarity(y.content),
    "resnik_brown": lambda x, y: resnik(x.content, y.content, brown_ic),
    "resnik_semcor": lambda x, y: resnik(x.content, y.content, semcor_ic),
    "hybrid": hybrid_wup
}

def similarity_test(word1, word2, pos=None, metric="cross", strategy="average", select1="max", select2="max", cheats=-1):
    #return cheats
    if not pos:
        pos = "nvar"
    elif type(pos) is tuple:
        pos = "nvar"
    else:
        pos = pos.lower()[0]
    metric = sim_metric[metric]
    strategy = sim_strategy[strategy]
    results = []
    for x in pos:
        word1_node = collects(strategy(word1, x))
        word2_node = collects(strategy(word2, x))
        if word1_node and word2_node:
            scores = collects([metric(a, b) for a in word1_node for b in word2_node])
            results.append(ListStrategy[select1](scores))
    if results:
        return ListStrategy[select2](results)
    return -1 # fallback of 0.5

def similarity_test2(word1, word2, pos, metric="cross", strategy="average", select1="max", select2="max", cheats=-1):
    if not pos:
        pos = ("nvars", "nvars")
    pos1, pos2 = pos
    metric = sim_metric[metric]
    strategy = sim_strategy[strategy]
    results = []
    for x in pos1:
        for y in pos2:
            word1_node = collects(strategy(word1, x))
            word2_node = collects(strategy(word2, y))
            if word1_node and word2_node:
                scores = collects([metric(x, y) for x in word1_node for y in word2_node])
                results.append(ListStrategy[select1](scores))
    if results:
        #print("{}\t{} -> ({}) | {}".format(word1, word2, ListStrategy[select2](results), results))
        return ListStrategy[select2](results)
    return -1 # fallback of 0.5



# In[7]:



def get_valid_scores(l1, l2):
    res1 = []
    res2 = []
    for x, y in zip(l1, l2):
        if y >= 0:
            res1.append(x)
            res2.append(y)
    print("dropping:", len(l1) - len(res1))
    return res1, res2

def confidence(rho, n):
    import math
    stderr = 1.0 / math.sqrt(n - 3)
    delta = 1.96 * stderr
    #print(rho, n, delta)
    lower = math.tanh(math.atanh(rho) - delta)
    upper = math.tanh(math.atanh(rho) + delta)
    return (lower, upper)


# In[43]:


SimExperiment = namedtuple("SimExperiment",
                           ["name", "data", "metric", "strategy", "select1", "select2", "pivot"]
                          )
SimTask = namedtuple("SimTask", ["word1", "word2", "gold", "pos"])
SimTaskResults = namedtuple("SimTaskResults", ["experiment", "instances", "spearman","pearson"])
