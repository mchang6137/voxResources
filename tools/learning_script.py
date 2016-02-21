import sys
import random
import scipy.io as spio
import numpy as np

import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import zero_one_loss
import sklearn.cross_validation as cv

import csv
import numpy as np

def evaluateLearner(name, learner, data, labels, splits):
    scores = cv.cross_val_score(learner, data, labels, cv = splits)
    print "-- %s Cross Validation Score --" % name
    print "Mean score: %f" % scores.mean()
    print "Standard deviation: %f" % scores.std()

feature_list = ['FV_MFCC.csv', 'FV_chroma.csv', 'FV_brightness.csv', 'FV_zerocross.csv']
fisher_directory = './../fisher_vectors/'
feature_data = []
SONG_SET_SIZE = 1000

for x in range(0,SONG_SET_SIZE):
    feature_data.append([])

#Parse CSV file, where each row corresponds to a particular exempler
#Concatenate all the features in the same list
for feature_file in feature_list:
    with open(fisher_directory + feature_file, 'rb') as f:
        reader = csv.reader(f)
        temp_feature_data = []
        for row in reader:
            float_row = [float(i) for i in row]
            temp_feature_data.append(float_row)
        
        count = 0
        for song_temp_feature in temp_feature_data:
            feature_data[count] += song_temp_feature
            count += 1

feature_data = np.array(feature_data)

maxLearners = 100
maxDepth = 5
N = len(feature_data)
labels = np.array([N,])

#Parse the labels file
with open(fisher_directory + 'labels.csv', 'rb') as f:
    reader = csv.reader(f)
    for label in reader:
        labels = np.array([int(i) for i in label])

rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)

# tune AdaBoost using grid search
boost = AdaBoostClassifier()
param_grid = [
    {'n_estimators': [25, 50, 75, 100], 'learning_rate': [0.1, 0.125, 0.25, 0.5, 0.75, 1]}
]
search = GridSearchCV(estimator = boost, param_grid = param_grid)
search.fit(feature_data, labels)

finalBoost = search.best_estimator_
boostParam = search.best_params_

evaluateLearner("Random Forest", rf, feature_data, labels, 10)
evaluateLearner(("Adaboost with %d Learners, %f Rate" % (boostParam['n_estimators'], boostParam['learning_rate'])),
    finalBoost, feature_data, labels, 10)
