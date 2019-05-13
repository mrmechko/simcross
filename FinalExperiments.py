#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from simcross.experiment import *
from simcross.data import *
from simcross.structs import *

e = ExperimentDefinition
def get_t_defs(name, f):
    name = name+f
    candidates = "average"
    return [
        #e(name, "resnik_brown", candidates, "max", "max", f),
        e(name, "normal_c", candidates, "max", "max", f),
        #e(name, "normal", candidates, "max", "max", f),
        #e(name, "resnik_trips", candidates, "max", "max", f),
        #e(name, "cross", candidates, "max", "max", f),
        #e(name, "pathlen", candidates, "max", "max", f),
        #e(name, "tripspath", candidates, "max", "max", f),
    ]
# In[5]:


df = None
_dcache={}

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

postuple=True

def run_MEN():
    print("running MEN")
    df = None
    test = MEN
    fname = ["dev", "test"]

    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("MEN-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("MEN.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)


# In[ ]:

def run_slex():
    print("running slex")
    df = None
    test = slex
    fname = ["n", "v", "vars", "nvars"]
    postuple=False
    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("slex-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("slex.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)

def run_verb():
    print("running sverb")
    df = None
    test = sverb
    fname = ["500-dev", "3000-test"]
    postuple=False
    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("slex-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("sverb.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)

# In[ ]:


def run_ws():
    print("running wordsim")
    df = None
    test = ws353
    fname = ["combined"]
    postuple=False

    definitions = []

    for f in fname:
        definitions += get_t_defs("ws353-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("ws.csv", 'w') as fl:
        df.sort_values(by=["name", "spearmanr"]).to_csv(fl)


#run_verb()
run_ws()
#run_slex()
#run_MEN()
