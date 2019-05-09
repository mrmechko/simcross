#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytrips.ontology import get_ontology as tripsont
from pytrips.structures import TripsType
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
from collections import defaultdict
from collections import namedtuple
import random

import math
from scipy.stats import spearmanr, pearsonr

nouns = list(wn.all_synsets(pos=wn.NOUN))
num = 100
noun_sample = random.sample(nouns, num)


# In[2]:


"""
Default node weights, can be overrided for variations
"""
_node_weights = defaultdict(lambda: 1)

_node_weights["fakeroot"] = 0

_dcache = {}


def _fallback(x, func, nonzero=False, fbs=-1):
    """
    Basic null checks for selecting a value from a list
    """
    if (len(x) == 0) or (sum(x) == 0 and nonzero):
        return fbs
    return func(x)

"""
Pick a value from a list
"""
ListStrategy = {
    'choose': lambda f: lambda b: lambda x: _fallback(x, f, fbs=-1),
    'min': lambda x: _fallback(x, min),
    'bmin': lambda b: lambda x: _fallback(x, min, fbs=b),
    'max': lambda x: _fallback(x, max),
    'bmax': lambda b: lambda x: _fallback(x, max, fbs=b),
    'average': lambda x: _fallback(x, lambda l: sum(l)/len(l), nonzero=True)
}


"""
Find the last common element between two lists
"""
def last_overlap(v1, v2, aligned=None):
    if not v1 or not v2:
        return aligned
    elif v1[0] != v2[0]:
        return aligned
    else:
        return last_overlap(v1[1:], v2[1:], aligned=v1[0])

class SemNode:
    def __init__(self, node):
        self._node = node

    def __eq__(self, other):
        if issubclass(type(other), SemNode):
            if type(other.content) == type(self.content):
                return other.content == self.content
        return False

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return "<SemNode: {}>".format(self.content.__repr__())

    def __str__(self):
        return "<SemNode: {}>".format(self.name)

    @property
    def name(self):
        """
        return canonical name for node
        """
        return str(self.content)

    @property
    def adjacent(self, label=None):
        """
        Return connected elements by label.  Useful for following non-hypernym relations in wordnet
        Can be used to implement other connections.
        Override as necessary.
        """
        return []

    @property
    def root(self):
        """
        Check if the node is a root of some sort.  Override as necessary for different resources.
        """
        return not self.parents

    @property
    def parents(self):
        """
        Get all parents for a node.  Perform any cross-resource cutoffs.
        """
        pass

    @property
    def children(self):
        """
        Get all children for a node.  Perform any cross resource cutoffs.
        """
        pass

    @property
    def content(self):
        """
        Return the wrapped node.
        """
        return self._node

    @property
    def resource(self):
        """
        Return the name of the relevant resource
        """
        return "default"

    def weight(self, weights=None):
        """
        Get resource-based weight of node.  Pass a weight dictionary as necessary.
        TODO: add an argument to pass individual weights
        """
        if not weights:
            weights = _node_weights
        return weights[self.resource]

    @staticmethod
    def path_depth(path, weights=None):
        """
        Get the total depth of a path, defined as a list of nodes.
        Does not check validity of path
        """

        return sum([p.weight(weights=weights) for p in path])

    @staticmethod
    def depth(node, weights=None, strategy='max'):
        """
        Get the depth of a node to a root using a weight dictionary and selection strategy.
        Default is minimum depth from any root.
        """
        if strategy in _dcache:
            if node.content in _dcache:
                return _dcache[strategy][node.content]
        else:
            _dcache[strategy] = {}
        weighted = [SemNode.path_depth(p, weights=weights) for p in node.paths_to_root()]
        _dcache[strategy][node.content] = ListStrategy[strategy](weighted) - 1
        return _dcache[strategy][node.content]

    @staticmethod
    def make(node):
        """
        Make a node based on the input type.
        Should add parameter dictionary to pass on to children
        """
        if type(node) is TripsType:
            return TripsNode(node)
        elif type(node) is Synset:
            return WordNetNode(node)
        elif type(node) is str:
            return WordNode(node)
        elif type(node) is SemNode:
            return node
        else:
            return None

    def paths_to_root(self):
        """
        Find all paths to a root based on hierarchy rules.
        Some resources return only one (Trips), others may return multiple (WordNet)
        """
        if self.root:
            return [[self]]
        res = []
        for c in self.parents:
            ptrs = c.paths_to_root()
            res.extend([t + [self] for t in ptrs if self not in t])
        return res

    def lcs_set(self, other):
        """
        Find the set of Lowest Common Subsumers for a node.
        Some resources have only one (Trips) other can have multiple (WordNet)
        """
        lcs = [last_overlap(p,q) for p in self.paths_to_root() for q in other.paths_to_root()]
        filtered = [x for x in lcs if x]
        if not filtered:
            return [TripsNode(tripsont()["root"])]
        return filtered

    def wupalmer(self, other, weights=None, depth_strategy='min', lcs_strategy='max'):
        """
        return cross-wupalmer measure using provided weights, depth_strategy and lcs_strategy
        depth_strategy: Choose max, min, or average depth over all paths
        lcs_strategy: Choose max, min, or average depth of lcs over all alternatives
        """
        if not issubclass(type(other), SemNode):
            other = SemNode.make(other) # this would break passing in an arbitrary maker object
        lcs_depth = ListStrategy[lcs_strategy]([SemNode.depth(d, weights, depth_strategy) for d in self.lcs_set(other)])
        sd = SemNode.depth(self, weights, depth_strategy)
        od = SemNode.depth(other, weights, depth_strategy)
        # nlwup = self.content.wup_similarity(other.content, simulate_root=True)
        return 2*(lcs_depth)/(sd + od)

    def path_similarity(self, other, weights=None, depth_strategy='max', lcs_strategy='max'):
        """
        Like wupalmer, except (d(s1) + d(s2) - 2 * lcs(s1,s2))
        """
        if not issubclass(type(other), SemNode):
            other = SemNode.make(other) # this would break passing in an arbitrary maker object
        lcs_depth = ListStrategy[lcs_strategy]([SemNode.depth(d, weights, depth_strategy) for d in self.lcs_set(other)])
        sd = SemNode.depth(self, weights, depth_strategy) + self.weight(weights=weights)
        od = SemNode.depth(other, weights, depth_strategy) + other.weight(weights=weights)
        return (sd + od) - 2 * lcs_depth


# In[3]:


class WordNode(SemNode):
    """
    Take a "word.pos" element as a node in the generalized hierarchy.
    """
    def resource(self):
        return "word"

    @property
    def name(self):
        return self.content

    @property
    def children(self):
        return []

    def word_pos(self):
        if "." in self._node:
            return self._node.split(".")
        return self._node, None

    @property
    def parents(self):
        """
        Lookup all TripsTypes, lookup all Wordnet Types.
        """
        w, p = self.word_pos()
        wordnet = wn.synsets(w, p)
        trips = tripsont().get_word(w, p)
        return [SemNode(c) for c in wordnet+trips]

class FakeRoot(SemNode):
    """
    FakeRoot for completeness purposes
    """
    def __init__(self):
        super(FakeRoot, self).__init__("fakeroot")

    def resource(self):
        return "fakeroot"

    @property
    def name(self):
        return "fakeroot"

    @property
    def parents(self):
        return []

    @property
    def children(self):
        return []


class TripsNode(SemNode):
    @property
    def name(self):
        return self.content.name

    @property
    def parents(self):
        return [SemNode.make(self._node.parent)]

    @property
    def children(self):
        return [SemNode.make(c) for c in self._node.children] + [SemNode.make(c) for c in self._node.wordnet]

    @property
    def resource(self):
        return "trips"

    @property
    def root(self):
        return self._node.depth == 0


class WordNetNode(SemNode):
    @property
    def name(self):
        return self.content.name()

    @property
    def parents(self):
        # NOTE: actually this is a little bit of a problem because we're not taking
        #       WN hypernyms
        tt = tripsont()[self._node]# + self._node.hypernyms()
        if not tt:
            tt = self._node.hypernyms()
        if not tt:
            return [FakeRoot()]
        return [SemNode.make(p) for p in tt]

    @property
    def children(self):
        return [SemNode.make(c) for c in self._node.hyponyms()]

    @property
    def resource(self):
        return "wordnet"



# # tests

# In[4]:


# equality

cat1 = wn.synset("cat.v.1")
cat2 = wn.synset("cat.v.1")

wcat1 = WordNetNode(cat1)
wcat2 = WordNetNode(cat2)

assert issubclass(type(wcat1), SemNode)
assert wcat1 == wcat2

# hypernyms

animal = SemNode.make(tripsont()["nonhuman-animal"])
mammal = SemNode.make(tripsont()["mammal"])
assert [mammal] == animal.parents


abbess = SemNode.make(wn.synset("abbess.n.1"))
scand  = SemNode.make(wn.synset("scandinavia.n.2"))
print(abbess.paths_to_root())
print(scand.paths_to_root())

wn.synset("scandinavia.n.2").hypernyms()


import pandas as pd

simlex = pd.read_csv("../SimLex-999/SimLex-999.txt", sep="\t")


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
            scores = collects([metric(x, y) for x in word1_node for y in word2_node])
            results.append(ListStrategy[select1](scores))
    if results:
        #print("{}\t{} -> ({}) | {}".format(word1, word2, ListStrategy[select2](results), results))
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

def ws353(name, metric, strategy, select1="max", select2="max", fname="combined"):
    wordsim = pd.read_csv("../wordsim353/{}.tab".format(fname), sep="\t")
    res = []
    for i, row in wordsim.iterrows():
        res.append(SimTask(row[0], row[1], row[2], None))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def slex(name, metric, strategy, select1="max", select2="max", fname="nvars"):
    wordsim = pd.read_csv("../SimLex-999/SimLex-999.txt".format(fname), sep="\t")
    res = []
    for i, row in wordsim.iterrows():
        if row[2].lower() in fname:
            res.append(SimTask(row[0], row[1], row[3], row[2].lower()))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def MEN(name, metric, strategy, select1="max", select2="max", fname="dev"):
    wordsim = pd.read_csv("../MEN/MEN_dataset_lemma_form.{}".format(fname), sep=" ")
    res = []
    for i, row in wordsim.iterrows():
        w1, w2, score = row[0][:-2], row[1][:-2], row[2],
        pos1, pos2 = get_pos_men(row[0][-1]), get_pos_men(row[1][-1])
        if pos1.lower() not in "n" and pos2.lower() not in "n":
            res.append(SimTask(w1, w2, score, (pos1, pos2)))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def MENNatural(name, metric, strategy, select1="max", select2="max", fname="dev"):
    wordsim = pd.read_csv("../MEN/MEN_dataset_natural_form_full", sep="\t")
    res = []
    for i, row in wordsim.iterrows():
        w1, w2, score = row[0], row[1], row[2],
        res.append(SimTask(w1, w2, score, None))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def MENIndiv(name, metric, strategy, select1="max", select2="max", fname="elias"):
    wordsim = pd.read_csv("../MEN/agreement/{}-men-ratings.txt".format(fname), sep=" ")
    res = []
    for i, row in wordsim.iterrows():
        w1, w2, score = row[0], row[1], row[2],
        res.append(SimTask(w1, w2, score, None))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def run_experiment(exp, postuple=False):
    results = []
    st = similarity_test
    if postuple:
        st = similarity_test2
    for d in exp.data:
        r = abs(exp.pivot - st(d.word1, d.word2, pos=d.pos,
                metric=exp.metric,
                strategy=exp.strategy,
                select1=exp.select1, select2=exp.select2,
                cheats=d.gold
               )) * 1/(1 - exp.pivot)
        g = d.gold
        if r >= 0:
            results.append((r,g))
    x = [r for r, g in results]
    y = [g for r, g in results]
    return SimTaskResults(exp, x, spearmanr(x, y), None)#pearsonr(x, y))

def experiment_string(exp, pandas=True, dataframe=None):
    columns = "name metric candidate choice1 choice2 instances spearmanr conf_low conf_high".split()
    if dataframe is None:
        dataframe = pd.DataFrame(columns=columns)
    e = exp.experiment
    if pandas:
        clow, chigh = confidence(exp.spearman.correlation, len(exp.instances))
        values = [e.name, e.metric, e.strategy, e.select1, e.select2, len(exp.instances),
        exp.spearman.correlation, clow, chigh] #exp.spearman.pvalue,
        #exp.pearson[0], exp.pearson[1]]
        res = pd.DataFrame({v: [r] for r, v in zip(values, columns)})
        return dataframe.append(res)
    return """
    ---
    name:      {}
    metric:    {}
    candidate: {}
    choice:    {}/{}
    ---
    instances:    {}
    spearman rho: {}
    p-value:      {}

    pearson rho:  {}
    p-value:      {}
    ================
    """.format(
        e.name, e.metric, e.strategy, e.select1, e.select2, len(exp.instances),
        exp.spearman.correlation, exp.spearman.pvalue,
        exp.pearson[0], exp.pearson[1]
    )



# In[9]:


# Tripswordnet
df = None
_dcache={}
fname = ["n", "v", "nv", "a"][0]

test = [ws353, slex][1]

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname)))

df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="average", select1="max", select2="max", fname=fname)), dataframe=df)

df = experiment_string(run_experiment(test("base", "normal", strategy="mfs", select1="max", select2="max", fname=fname)), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="mfs", select1="max", select2="max", fname=fname)), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="lookup", select1="max", select2="max", fname=fname)), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="both", select1="max", select2="max", fname=fname)), dataframe=df)



df


# In[10]:


# Tripswordnet
df = None
_dcache={}
fname = ["n", "v", "nv", "vars", "a", "nvars"][3]

test = [ws353, slex][1]

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname)))

df = experiment_string(run_experiment(test("Trips-Wordnet", "hybrid", strategy="average", select1="max", select2="max", fname=fname)), dataframe=df)

#df = experiment_string(run_experiment(test("base", "normal", strategy="mfs", select1="max", select2="max", fname=fname)), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-av", "cross", strategy="average", select1="max", select2="max", fname=fname)), dataframe=df)

#df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="lookup", select1="max", select2="max", fname=fname)), dataframe=df)

#df = experiment_string(run_experiment(test("Trips-Wordnet", "cross", strategy="both", select1="max", select2="max", fname=fname)), dataframe=df)
fname = ["n", "v", "nv", "vars", "a", "nvars"][1]
df = experiment_string(run_experiment(test("Trips-Wordnet-v", "cross", strategy="average", select1="max", select2="max", fname=fname)), dataframe=df)



df


# In[44]:


# Tripswordnet
df = None
#_dcache={}
fname = ["test", "dev"][1]

test = [ws353, slex, MEN][2]

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

postuple = True

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple))

#df = experiment_string(run_experiment(test("Trips-Wordnet", "hybrid", strategy="average", select1="max", select2="max", fname=fname), postuple=True), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-pt", "cross", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="both", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="lookup", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)


df


# In[40]:


# Tripswordnet
df = None
#_dcache={}
fname = ["test", "dev"][0]

test = [ws353, slex, MEN][2]

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

postuple = True

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple))

#df = experiment_string(run_experiment(test("Trips-Wordnet", "hybrid", strategy="average", select1="max", select2="max", fname=fname), postuple=True), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-pt", "cross", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="both", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="lookup", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df


# In[15]:


# Tripswordnet
df = None
_dcache={}
fname = ["test", "dev"][0]

test = [ws353, slex, MEN, MENNatural][3]

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname), postuple=False))

#df = experiment_string(run_experiment(test("Trips-Wordnet", "hybrid", strategy="average", select1="max", select2="max", fname=fname), postuple=True), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-pt", "cross", strategy="average", select1="max", select2="max", fname=fname), postuple=False), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="both", select1="max", select2="max", fname=fname), postuple=False), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="lookup", select1="max", select2="max", fname=fname), postuple=False), dataframe=df)




df


# In[32]:


# Tripswordnet
df = None
_dcache={}
fname = ["elias", "marcos"][0]

test = MENIndiv
postuple=True

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

df = experiment_string(run_experiment(test("base", "normal", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple))

#df = experiment_string(run_experiment(test("Trips-Wordnet", "hybrid", strategy="average", select1="max", select2="max", fname=fname), postuple=True), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-pt", "cross", strategy="average", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="both", select1="max", select2="average", fname=fname), postuple=postuple), dataframe=df)

df = experiment_string(run_experiment(test("Trips-Wordnet-npt", "cross", strategy="lookup", select1="max", select2="max", fname=fname), postuple=postuple), dataframe=df)


df


# In[ ]:
