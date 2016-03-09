
# coding: utf-8

import numpy as np
import matplotlib.pylab as pl
from numpy import *
from matplotlib.pylab import *

figure(figsize=(8*2, 6*2))
hold(True)

T = 20.28
#T = 20.272
path = '/home/pavel/projects/hh_cuda/phase_map_T_{:.3f}_{:n}per.npy'

dts, dtPost = load(path.format(T, 1))
plot(dts, dtPost, label='1 period')

dts, dtPost = load(path.format(T, 3))
plot(dts, dtPost, label='3 period')

dts, dtPost = load(path.format(T, 9))
plot(dts, dtPost, label='9 period')

plot(arange(0, T, 0.02), arange(0, T, 0.02), '--g')
xlabel(r'$\varphi_n$', fontsize=24.0); ylabel(r'$\varphi_{n+1}$', fontsize=24.0)
gca().tick_params(axis='both', which='major', labelsize=24.)
xlim((0, T)); ylim((0, T))
legend(loc='upper left')
