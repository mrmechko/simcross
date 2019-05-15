import pandas as pd
from .structs import *
#from .resnik import trips_resnik, wn_trips_resnik, wn_resnik2
from nltk.corpus import wordnet_ic

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

def can_wup(sn1, sn2, f):
    if n_or_v2(sn1.content) and sn1.content.pos() == sn2.content.pos():
        return f(sn1, sn2)
    return -1

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

def _list_fallback(func, fallback_func, args, both=False):
    result = func(*args)
    if not result or both:
        result += fallback_func(*args)
    return result

def wn_pl(x, y):
    res = x.content.path_similarity(y.content)
    if res or res == 0:
        return res
    return float("-inf")

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
    "normal_c" : lambda x, y: x.wupalmer(y, node_weights, use_trips=False),
    "tripspath" : lambda x, y: x.path_similarity(y, weights=node_weights),
    "pathlen" : wn_pl,
    "normal": lambda x, y: can_wup(x, y, lambda a, b: a.content.wup_similarity(b.content, simulate_root=True)),
    #"resnik_brown":  lambda x, y: can_wup(x, y, wn_resnik2),
    #"resnik_trips": wn_trips_resnik,
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
