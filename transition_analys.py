'''
Created on Feb 17, 2015

@author: Esir Pave
'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter as gs_filter
import matplotlib.pylab as pl

def analys_trans(fname, maxSr=15, srHstBins=10):
    srHstBinSize = maxSr/srHstBins

    (time, sr) = np.load(fname)
    sr = gs_filter(sr, 1)

    srHst = np.histogram(sr, bins=srHstBins, range=(0, maxSr))[0]

    srMidpntIdx = len(srHst)/2
    firstMaxIdx = np.argmax(srHst[:srMidpntIdx])
    secondMaxIdx = srMidpntIdx + np.argmax(srHst[srMidpntIdx:])
    srMidpnt = np.argmin(srHst[firstMaxIdx:secondMaxIdx])*srHstBinSize
#     srMidpnt = (firstMaxIdx + secondMaxIdx)/2
#    print(srMidpnt)
    srMidpnt = 3.68

    thr = np.array(sr >= srMidpnt, dtype='int32')
    df = np.diff(thr)
    indices_up = pl.find(df == 1)
    indices_down = pl.find(df == -1)

    TimesUp = []
    TimesDown = []
    Periods = []
    # all time in sec
    if sr[0] > srMidpnt:
        for up, down in zip(indices_up, indices_down):
            TimesDown.append(up - down)
        for up, down in zip(indices_up, indices_down[1:]):
            TimesUp.append(down - up)
        Periods = np.diff(indices_down)
    elif sr[0] <= srMidpnt:
        for up, down in zip(indices_up, indices_down):
            TimesUp.append(down - up)
        for up, down in zip(indices_up[1:], indices_down):
            TimesDown.append(up - down)
        Periods = np.diff(indices_up)
    else:
        raise Exception("Unexpected type of sr")
    return np.array(Periods), np.array(TimesDown), np.array(TimesUp)
