# python3.7
# -*- coding: utf-8 -*-
"""
@author: machengnan
@contact:2624079968@qq.com
@version: 1.0.0
@file: mfcc_example.py
@time: 2020/4/26 11:31
"""

import matplotlib.pyplot as plt
import librosa.display
import librosa
#import IPython.display as ipd

y1, sr1 = librosa.load('./sounds/10.wav')
y2, sr2 = librosa.load('./sounds/78.wav')

# plt.subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
# librosa.display.specshow(mfcc1)
# plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
# librosa.display.specshow(mfcc2)
print(mfcc1.shape)
print(mfcc2.shape)


from dtw import dtw
from numpy.linalg import norm
dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)


plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))
plt.show()