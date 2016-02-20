import sys
import random
import scipy.io as spio
import numpy as np

import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import sklearn.cross_validation as cv

import csv
import numpy as np

exempler = []

#Parse CSV file, wher each row corresponds to a particular exempler
with open('FV_MFCC.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        float_row = np.array([float(i) for i in row])
        exempler.append(float_row)

exempler = np.array(exempler)

maxLearners = 100
maxDepth = 5
N = len(exempler)
labels = np.array([N,])

#Parse the labels file
with open('labels.csv', 'rb') as f:
    reader = csv.reader(f)
    for label in reader:
        labels = np.array([int(i) for i in label])

TEidx = np.array(random.sample(range(0,N), N/10))
X_TE = exempler[TEidx, :]
X_TR = exempler[[i for i in range(0,N) if i not in TEidx],:]
Y_TE = labels[TEidx]
Y_TR = labels[[i for i in range(0,N) if i not in TEidx]]

rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)
scores = cv.cross_val_score(rf, exempler, labels, cv=5)
print "Cross validation scores: " + str(scores)
print "Mean: %f" % scores.mean()
print "Standard deviation: %f" % scores.std()
