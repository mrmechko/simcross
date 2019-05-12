#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from simcross.experiment import *
from simcross.data import *
from simcross.struct import *


# In[5]:


df = None
_dcache={}

node_weights["fakeroot"] = 1
node_weights["wordnet"] = 1
node_weights["trips"] = 1
node_weights["word"] = 1

postuple=True


# In[6]:



fname = ["elias", "marcos"][0]

test = MENIndiv
postuple=True

e = ExperimentDefinition
definitions = [
    e("MEN-elias", "resnik_brown", "average", "max", "max", fname),
    e("MEN-elias", "resnik_trips", "average", "max", "max", fname),
]

for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)


# In[ ]:


fname = ["elias", "marcos"][1]



e = ExperimentDefinition
definitions = [
    e("MEN-marcos", "resnik_brown", "average", "max", "max", fname),
    e("MEN-marcos", "resnik_trips", "average", "max", "max", fname),
]

for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)


# In[ ]:


test = MENNatural
fname = ["full"]

e = ExperimentDefinition
definitions = []

for f in fname:
    definitions += [
        e("MENNatural-"+f, "resnik_brown", "average", "max", "max", f),
        e("MENNatural-"+f, "resnik_trips", "average", "max", "max", f)
    ]


for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)


# In[ ]:


test = MEN
fname = ["dev", "test"]

e = ExperimentDefinition
definitions = []

for f in fname:
    definitions += [
        e("MEN-"+f, "resnik_brown", "average", "max", "max", f),
        e("MEN-"+f, "resnik_trips", "average", "max", "max", f),
        e("MEN-"+f, "resnik_trips", "lookup", "max", "max", f),
        e("MEN-"+f, "resnik_trips", "mfs", "max", "max", f)
    ]


for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)


# In[ ]:


MEN_res = df
MEN_res.sort_values(by="spearmanr")


# In[ ]:


df = None
test = slex
fname = ["n", "v", "vars", "nvars"]
postuple=False
e = ExperimentDefinition
definitions = []

for f in fname:
    definitions += [
        e("slex-"+f, "resnik_brown", "average", "max", "max", f),
        e("slex-"+f, "resnik_trips", "average", "max", "max", f),
#       e("slex-"+f, "resnik_trips", "lookup", "max", "max", f),
#       e("slex-"+f, "resnik_trips", "mfs", "max", "max", f)
    ]


for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

df
slex_res = df
slex_res.sort_values(by="spearmanr")


# In[ ]:


df = None
test = ws353
fname = ["combined", "relatedness", "similarity"]
postuple=False
e = ExperimentDefinition
definitions = []

for f in fname:
    definitions += [
        e("ws353-"+f, "resnik_brown", "average", "max", "max", f),
        e("ws353-"+f, "resnik_trips", "average", "max", "max", f),
#        e("ws353-"+f, "resnik_trips", "average", "max", "average", f),
#        e("ws353-"+f, "resnik_trips", "mfs", "max", "max", f)
    ]


for t in definitions:
    df = experiment_string(run_experiment(run_dataset(t, test), postuple=postuple), dataframe=df)

df

ws_res = df
ws_res.sort_values(by="spearmanr")


# In[ ]:



d = [("ws.csv", ws_res), ("slex.csv", slex_res), ("men.csv", MEN_res)]

for f, y in d:
    with open(f, 'w') as fl:
        y.to_csv(fl)
