# python3.7
# -*- coding: utf-8 -*-
"""
@author: machengnan
@contact:2624079968@qq.com
@version: 1.0.0
@file: knn.py
@time: 2020/4/26 11:31
"""

import time
import os
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from dtw import dtw
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier


dirname = "train"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]
start = time.time()
# minval = 200
distances = np.ones((len(files), len(files)))
y = np.ones(len(files))

for i in range(len(files)):
    y1, sr1 = librosa.load(dirname+"/"+files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    for j in range(len(files)):
        y2, sr2 = librosa.load(dirname+"/"+files[j])
        mfcc2 = librosa.feature.mfcc(y2, sr2)
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
#         print files[i], mfcc1.T[0][0], mfcc2.T[0][0], files[j], dist
#         if dist < minval:
#             minval = dist
        distances[i,j] = dist
    if i % 2 == 0:
        y[i] = 0  # 'a'
    else:
        y[i] = 1  # 'b'

end = time.time()
print("Time used: {}s".format(end-start))
label = ['a','b']
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier.fit(distances, y)

y, sr = librosa.load('./test/farw0-b1-t.wav')
mfcc = librosa.feature.mfcc(y, sr)
distanceTest = []
for i in range(len(files)):
    y1, sr1 = librosa.load(dirname+"/"+files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: norm(x - y, ord=1))
    distanceTest.append(dist)

# print(distanceTest)
pre = classifier.predict([distanceTest])[0]
print("Predict audio is: '{}'".format(label[int(pre)]))