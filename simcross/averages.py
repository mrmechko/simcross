import logging
logging.basicConfig(level=logging.CRITICAL)

#from nltk.probability import ConditionalFreqDist
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic
from .structs import SemNode, WordNetNode
from .resnik import wn_trips
from .experiment import similarity_test
from collections import Counter, defaultdict
from pytrips.ontology import load as load_ont
import statistics
from scipy.stats import spearmanr

ont = load_ont()

posmap = defaultdict(lambda: "X")
posmap["NOUN"] = "n"
posmap["VERB"] = "v"
posmap["ADJ"] = "ars"
posmap["ADV"] = "ars"

words = brown.tagged_words(tagset="universal")
import random

sample_size = 5000
nouns = random.sample([w for w in words if w[1] == "NOUN"], sample_size)
verbs = random.sample([w for w in words if w[1] == "VERB"], sample_size)

def compare_depth(word, pos, verbose=False):
    t_depths = [x.depth(x) for x in set(wn_trips(word, pos))]
    w_depths = [c.min_depth() for c in wn.synsets(word, posmap[pos])]
    return t_depths, w_depths

def compare_wup(sample, iters=50):
    fn = lambda w1, w2, p: similarity_test(w1, w2, p, metric="normal")
    return [
        similarity_test(s, t, p)
        for s, p in random.sample(sample, iters) for t in random.sample(sample, iters)
    ]


def write_wup(iters):
    n = compare_wup(nouns, iters)
    v = compare_wup(verbs, iters)
    print("mean:\n\tnouns:\t{}\n\tverbs\t{}".format(
        statistics.mean(n), statistics.mean(v)
    ))
    print("stdev:\n\tnouns:\t{}\t{}\n\tverbs\t{}\t{}".format(
        statistics.stdev(n), statistics.stdev(v),
    ))

def compare_pos(samples):
    trips_depths_max, trips_depths_min = [0], [0]
    wn_depths_max, wn_depths_min = [0], [0]
    for n, p in samples:
        t, w = compare_depth(n, p)
        trips_depths_max.append(max(t+[0]))
        wn_depths_max.append(max(w+[0]))
        trips_depths_min.append(min(t+[max(trips_depths_max)]))
        wn_depths_min.append(min(w+[max(wn_depths_max)]))
    print("mean:\n\ttrips:\t{}\t{}\n\twn   \t{}\t{}".format(
        statistics.mean(trips_depths_max), statistics.mean(trips_depths_min),
        statistics.mean(wn_depths_max), statistics.mean(wn_depths_min)
    ))
    print("stdev:\n\ttrips:\t{}\t{}\n\twn   \t{}\t{}".format(
        statistics.stdev(trips_depths_max), statistics.stdev(trips_depths_min),
        statistics.stdev(wn_depths_max),
        statistics.stdev(wn_depths_min)
    ))

    print("spearman:\n\tMAX: {}\n\tMIN: {}".format(
            spearmanr(trips_depths_max, wn_depths_max),
            spearmanr(trips_depths_min, wn_depths_min)
    ))
