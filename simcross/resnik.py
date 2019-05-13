import logging
logging.basicConfig(level=logging.CRITICAL)

from pytrips.ontology import load
from nltk.probability import ConditionalFreqDist
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic
from .structs import SemNode, WordNetNode
from collections import Counter, defaultdict

brown_ic = wordnet_ic.ic("ic-semcorraw-resnik-add1.dat")
ont = load()

posmap = defaultdict(lambda: "X")
posmap["NOUN"] = "n"
posmap["VERB"] = "v"
posmap["ADJ"] = "ars"
posmap["ADV"] = "ars"

def q(w, p):
    if not w.startswith("q::"):
        w = "q::"+w
    if posmap[p.upper()] != "X":
        p = posmap[p.upper()]
    return (w, p)

def node_getter(word, pos):
    return []

def wordnet_node(word, pos):
    return [SemNode.make(p, False) for p in wn.synsets(word, pos)]

def trips_only(word, pos):
    return [SemNode.make(n) for n in ont.get_word(word, pos)]

def wn_trips(word, pos):
    r = ont[q(word, pos)]
    return [SemNode.make(n) for n in set(r["lex"] + r["wn"])]

def make_CPD(get_nodes):
    cpd = ConditionalFreqDist([(q(w, p)[1], w) for w, p in brown.tagged_words(tagset="universal")])

    ctr = defaultdict(Counter)

    for pos in ["n", "v", "ars"]:
        total = cpd[pos].N()
        ctr_ = ctr[pos]
        for w, c in cpd[pos].items():
            nodes_of_interest = get_nodes(w, pos)
            for n in nodes_of_interest:
                s = [n]
                seen = set()
                while s:
                    r = s.pop(0)
                    if r in seen:
                        continue
                    seen.add(r)
                    ctr_[r] += c
                    s.extend(r.parents)
        for w in cpd[pos]:
           cpd[pos][w] = math.log(cpd[pos][w]) - math.log(total)
    final_ctr = Counter()
    for pos in ["n", "v", "ars"]:
        for s,v in ctr[pos].items():
            if v != 0:
                if final_ctr[s] == 0:
                    final_ctr[s] = v
                else:
                    final_ctr[s] += max([v, final_ctr[s]])
    return final_ctr

import math
def maxs(v, default=float("-inf")):
    for a in v:
        if a > default:
            default = a
    return default

def resnik(cpd, node1, node2, full=False, nodes=False):
    if nodes:
        node1 = SemNode.make(node1)
        node2 = SemNode.make(node2)
    #print(type(cpd))
    res = {l: -cpd[l] for l in node1.lcs_set(node2) if cpd[l]}
    if full:
        return res
    return maxs(res.values())

def resnik_w(cpd, q1, q2, use_wn=True, use_trips=True):
    r1 = ont[q1]
    r2 = ont[q2]
    return max([resnik(cpd, a, b, False, True) for a in r1["lex"] + r1["wn"] for b in r2["lex"] + r2["wn"]])

def wn_resnik(q1, q2):
    return max([s1.res_similarity(s2, ic=brown_ic) for s1 in wn.synsets(q1[0][3:], q1[1]) for s2 in wn.synsets(q2[0][3:], q2[1])])

def wn_resnikq(q1, q2):
    try:
        return q1.content.res_similarity(q2.content, ic=brown_ic)
    except:
        return 0


wn_only_cpd = make_CPD(wordnet_node)
trips_only_cpd = make_CPD(trips_only)
wn_trips_only_cpd = make_CPD(wn_trips)

def check(q1, q2):
    print(q1, q2)

    print("resnik:", wn_resnik(q1, q2))
    print("wn_only:", resnik_w(wn_only_cpd, q1, q2))
    print("trips_only:", resnik_w(trips_only_cpd, q1, q2))
    print("wn_trips_only:", resnik_w(wn_trips_only_cpd, q1, q2))

# check(q("cat", "n"), q("fish", "n"))
# check(q("dog", "n"), q("fish", "n"))
# check(q("bread", "n"), q("fish", "n"))
# check(q("bread", "n"), q("dog", "n"))
# check(q("cake", "n"), q("bread", "n"))

def wn_resnik2(x, y):
    if x.content.pos() in [wn.NOUN, wn.VERB] and y.content.pos() in [wn.NOUN, wn.VERB]:
        return x.content.res_similarity(y.content, brown_ic)
    else:
        return -1
    if type(x) is not WordNetNode or type(y) is not WordNetNode:
        print("sup")
        return -1
    x.use_trips = False
    y.use_trips = False
    #print(type(wn_only_cpd))
    return resnik(wn_only_cpd, x, y)

trips_resnik = lambda x, y: resnik(trips_only_cpd, x, y)
wn_trips_resnik = lambda x, y: resnik(wn_trips_only_cpd, x, y)
