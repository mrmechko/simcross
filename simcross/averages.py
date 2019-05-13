import logging
logging.basicConfig(level=logging.CRITICAL)

#from nltk.probability import ConditionalFreqDist
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic
from .structs import SemNode, WordNetNode
from .resnik import wn_trips
from collections import Counter, defaultdict
from pytrips.ontology import load as load_ont

ont = load_ont()

posmap = defaultdict(lambda: "X")
posmap["NOUN"] = "n"
posmap["VERB"] = "v"
posmap["ADJ"] = "ars"
posmap["ADV"] = "ars"

words = brown.tagged_words(tagset="universal")
import random

sample_size = 1000
nouns = random.sample([w for w in words if w[1] == "NOUN"], sample_size)
verbs = random.sample([w for w in words if w[1] == "VERB"], sample_size)

def compare_depth(word, pos, verbose=False):
    t_depths = [x.depth(x) for x in set(wn_trips(word, pos))]
    w_depths = [c.min_depth() for c in wn.synsets(word, posmap[pos])]
    if verbose:
        print(word, pos)
        print(t_depths)
        print(w_depths)
        print("-------")
    return t_depths, w_depths


def compare_pos(samples):
    trips_depths = []
    wn_depths = []
    for n, p in samples:
        t, w = compare_depth(n, p)
        trips_depths.extend(t)
        wn_depths.extend(w)
    print("averages:\n\ttrips:\t{}\n\twn   \t{}".format(
        sum(trips_depths)/len(trips_depths),
        sum(wn_depths)/len(wn_depths)
    ))
    print("unique values:\n\ttrips:\t{}\n\twn   \t{}".format(
        len(set(trips_depths)),
        len(set(wn_depths))
    ))
