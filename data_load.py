# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:11:51 2015

@author: pavel
"""

import numpy as np
import csv

def oscill_load(path):
    t = []
    Vm = []
    f = open(path, "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        t.append(l[0])
        Vm.append(l[1])
    f.close()

    t = np.array(t[1:], dtype='float32')
    Vm = np.array(Vm[1:], dtype='float32')
    return t, Vm

def loadSpkTimes(path):
    f = open(path , "r")

    rdr = csv.reader(f,delimiter="\t")

    snd = []
    tm = []
    for l in rdr:
        snd.append(l[0])
        tm.append(l[1])
    f.close()
    tm = np.array(tm, dtype="float32")
    snd = np.array(snd, dtype="float32")

    return tm, snd