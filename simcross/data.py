import pandas as pd
from collections import defaultdict
from collections import namedtuple
import random

import math
from scipy.stats import spearmanr, pearsonr

from .experiment import *
from .struct import *

ExperimentDefinition = namedtuple("Experiment", ["name", "metric", "strategy", "select1", "select2", "fname"])

def run_dataset(edef, dataset):
    return dataset(edef.name, edef.metric, edef.strategy, edef.select1, edef.select2, edef.fname)


def ws353(name, metric, strategy, select1="max", select2="max", fname="combined"):
    wordsim = pd.read_csv("wordsim353/{}.tab".format(fname), sep="\t")
    res = []
    for i, row in wordsim.iterrows():
        res.append(SimTask(row[0], row[1], row[2], None))
    print(len(res))
    pivot=0
    if fname == "relatedness":
        pivot=0.7
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def slex(name, metric, strategy, select1="max", select2="max", fname="nvars"):
    wordsim = pd.read_csv("SimLex-999/SimLex-999.txt".format(fname), sep="\t")
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
    wordsim = pd.read_csv("MEN/MEN_dataset_lemma_form.{}".format(fname), sep=" ")
    res = []
    for i, row in wordsim.iterrows():
        w1, w2, score = row[0].split("-")[0], row[1].split("-")[0], row[2],
        pos1, pos2 = get_pos_men(row[0][-1]), get_pos_men(row[1][-1])
        res.append(SimTask(w1, w2, score, (pos1, pos2)))
    print(len(res))
    pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def MENNatural(name, metric, strategy, select1="max", select2="max", fname="dev"):
    wordsim = pd.read_csv("MEN/MEN_dataset_natural_form_full", sep=" ")
    res = []
    for i, row in wordsim.iterrows():
        w1, w2, score = row[0], row[1], row[2],
        res.append(SimTask(w1, w2, score, None))
    print(len(res))
    pivot=0
    return SimExperiment(name, res, metric, strategy, select1, select2, pivot)

def MENIndiv(name, metric, strategy, select1="max", select2="max", fname="elias"):
    wordsim = pd.read_csv("MEN/agreement/{}-men-ratings.txt".format(fname), sep=" ")
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
