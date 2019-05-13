import logging
logging.basicConfig(level=logging.CRITICAL)

from pytrips.ontology import get_ontology as tripsont
from pytrips.structures import TripsType
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import wordnet as wn

from collections import defaultdict



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
        _dcache[strategy][node.content] = ListStrategy[strategy](weighted) #- node.weight(weights=weights)
        return _dcache[strategy][node.content]

    @staticmethod
    def make(node, use_trips=True):
        """
        Make a node based on the input type.
        Should add parameter dictionary to pass on to children
        """
        if type(node) is TripsType:
            print(node)
            return TripsNode(node)
        elif type(node) is Synset:
            return WordNetNode(node, use_trips)
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

    def wupalmer(self, other, weights=None, depth_strategy='max', lcs_strategy='max', use_trips=True):
        """
        return cross-wupalmer measure using provided weights, depth_strategy and lcs_strategy
        depth_strategy: Choose max, min, or average depth over all paths
        lcs_strategy: Choose max, min, or average depth of lcs over all alternatives
        """
        if not issubclass(type(other), SemNode):
            other = SemNode.make(other, use_trips=use_trips) # this would break passing in an arbitrary maker object
        elif type(other) is WordNetNode and type(self) is WordNetNode:
            other.use_trips = use_trips
            self.use_trips = use_trips
        lcs_depth = ListStrategy[lcs_strategy]([SemNode.depth(d, weights, depth_strategy) for d in self.lcs_set(other)])
        sd = SemNode.depth(self, weights, depth_strategy)
        od = SemNode.depth(other, weights, depth_strategy)
        return 2*(lcs_depth)/(sd + od)

    def path_similarity(self, other, weights=None, depth_strategy='max', lcs_strategy='max', use_trips=True):
        """
        Like wupalmer, except (d(s1) + d(s2) - 2 * lcs(s1,s2))
        """
        if not issubclass(type(other), SemNode):
            other = SemNode.make(other, use_trips=use_trips) # this would break passing in an arbitrary maker object
        elif type(other) is WordNetNode and type(self) is WordNetNode:
            other.use_trips = use_trips
            self.use_trips = use_trips
        lcs_depth = ListStrategy[lcs_strategy]([SemNode.depth(d, weights, depth_strategy) for d in self.lcs_set(other)])
        sd = SemNode.depth(self, weights, depth_strategy)
        od = SemNode.depth(other, weights, depth_strategy)
        return 1/(1+(sd + od) - 2 * lcs_depth)


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
    def __init__(self, content, use_trips):
        super(WordNetNode, self).__init__(content)
        self.use_trips = use_trips

    @property
    def name(self):
        return self.content.name()

    @property
    def parents(self):
        # NOTE: actually this is a little bit of a problem because we're not taking
        #       WN hypernyms
        tt = set()
        if self.use_trips:
            tt = set(tripsont()[self._node])# + self._node.hypernyms()
        if not tt:
            ntt = self._node.hypernyms()
            for s in ntt:
                if not set(tripsont()[s]).intersection(tt):
                    tt.add(s)
        if not tt:
            return [FakeRoot()]
        return [SemNode.make(p, self.use_trips) for p in tt]

    @property
    def children(self):
        return [SemNode.make(c) for c in self._node.hyponyms()]

    @property
    def resource(self):
        return "wordnet"
