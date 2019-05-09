from simcross.struct import SemNode
from pytrips.ontology import load
from nltk.corpus import wordnet as wn

ont = load()

bread_trips = ont["ont::bread"]
muffin_wordnet = wn.synset("muffin.n.1")

bread_sn = SemNode.make(bread_trips)
muffin_sn = SemNode.make(muffin_wordnet)


print(bread_sn, "->", muffin_sn, bread_sn.wupalmer(muffin_sn))
