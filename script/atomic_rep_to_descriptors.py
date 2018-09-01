#!/usr/bin/env python
import numpy as np

def atomic_rep_to_compound_descriptor(rep, mom_order=1, icov=False):

    if (rep.ndim == 1):
        rep = np.reshape(rep, (-1, rep.shape[0]))

    d = []
    d.extend(np.mean(rep, axis=0))
    if (mom_order > 1):
        d.extend(np.std(rep, axis=0))
    if (mom_order > 2):
        d.extend(sp.stats.skew(rep, axis=0))
    if (mom_order > 3):
        d.extend(sp.stats.kurtosis(rep, axis=0))
    if (icov == True):
        if (rep.shape[0] == 1):
            cov = np.zeros((rep.shape[1], rep.shape[1]))
        else:
            cov = np.cov(rep.T)
        for i,row in enumerate(cov):
            for j,val in enumerate(row):
                if (i<j):
                    d.append(val)

    return np.array(d)
