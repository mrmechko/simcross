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
        e(name, "cross", candidates, "max", "max", f),
        #e(name, "resnik_brown", candidates, "max", "max", f),
        #e(name, "normal_c", candidates, "max", "max", f),
        e(name, "normal", candidates, "max", "max", f),
        #e(name, "resnik_trips", candidates, "max", "max", f),
        e(name, "pathlen", candidates, "max", "max", f),
        e(name, "tripspath", candidates, "max", "max", f),
    ]
# In[5]:


df = None
_dcache={}

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

postuple=True

def run_MEN(df=None):
    print("running MEN")
    test = MEN
    fname = ["dev", "test"]

    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("MEN-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=False), dataframe=df)

    with open("MEN.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)
    return df


# In[ ]:

def run_slex(df=None):
    print("running slex")
    test = slex
    fname = ["nvars"] #, "vars", "nvars"]
    postuple=False
    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("slex-", f)

    for t in definitions:
        df = experiment_string(run_experiment_to_store(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("slex.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)
    return df

def run_verb(df=None):
    print("running sverb")
    test = sverb
    fname = ["3000-test"]
    postuple=False
    e = ExperimentDefinition
    definitions = []

    for f in fname:
        definitions += get_t_defs("slex-", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("sverb.csv", 'w') as fl:
        df.sort_values(by="spearmanr").to_csv(fl)
    return df

# In[ ]:


def run_ws(df=None):
    print("running wordsim")
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
    return df

def run_smalls(df=None):
    print("running smalls")
    test = smalls
    fname = ["MC", "RG"]
    postuple=False

    definitions = []

    for f in fname:
        definitions += get_t_defs("", f)

    for t in definitions:
        df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

    with open("smalls.csv", 'w') as fl:
        df.sort_values(by=["name", "spearmanr"]).to_csv(fl)
    return df


#df = run_verb()
#df = run_ws(df)
df = run_slex(df)
#df = run_MEN(df)
#df = run_smalls(df)


with open("all.csv", 'w') as fl:
    df.sort_values(by=["name", "spearmanr"]).to_csv(fl)
