import sys
import random
import scipy.io as spio
import numpy as np

import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
import sklearn.cross_validation as cv

import csv
import numpy as np

def evaluateLearner(name, learner, data, labels, trials):
    scores = cv.cross_val_score(learner, data, labels, cv = trials)
    print "-- %s Cross Validation Score --" % name
    print "Mean score: %f" % scores.mean()
    print "Standard deviation: %f" % scores.std()

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

rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)
boost1 = AdaBoostClassifier(n_estimators = maxLearners)

evaluateLearner("Random Forest", rf, exempler, labels, 5)
evaluateLearner("Default AdaBoost", boost1, exempler, labels, 5)
