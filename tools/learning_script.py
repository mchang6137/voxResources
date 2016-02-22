import sys
import random
import scipy.io as spio
import numpy as np

import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import sklearn.cross_validation as cv

import csv
import numpy as np

# does grid search with the given learner, data, and labels on the
# specified param grid, prints best params, returns the best learner 
# produced by the search
def tuneLearner(name, learner, param_grid, data, labels):
    search = GridSearchCV(estimator = learner, param_grid = param_grid)
    search.fit(data, labels)
    print "-- %s Grid Search Parameters --" % name
    for param in search.best_params_:
        print "%s: %s" % (param, str(search.best_params_[param]))
    print "--\n"
    return search.best_estimator_

def evaluateLearner(name, learner, data, labels, splits):
    scores = cv.cross_val_score(learner, data, labels, cv = splits)
    print "-- %s Cross Validation Score --" % name
    print "Mean score: %f" % scores.mean()
    print "Standard deviation: %f" % scores.std()
    print "--\n"

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

# tune random forest using grid search and evaluate
rf = RandomForestClassifier(warm_start = False)
rf_param_grid = [
    {'n_estimators': [10, 25, 50, 75, 100], 'max_depth': [4, 5, 6, 7]}
]
final_rf = tuneLearner("Random Forest", rf, rf_param_grid, feature_data, labels)
evaluateLearner("Random Forest", rf, feature_data, labels, 10)

# tune AdaBoost using grid search
#boost = AdaBoostClassifier()
#boost_param_grid = [
    #{'n_estimators': [25, 50, 75, 100], 'learning_rate': [0.1, 0.125, 0.25, 0.5, 0.75, 1]}
#]
#final_boost = tuneLearner("Adaboost", boost, boost_param_grid, feature_data, labels)
#evaluateLearner("Adaboost", final_Boost, feature_data, labels, 10)

# tune SVM using grid search and evaluate
svm = SVC()
svm_param_grid = [
    {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75], 'kernel': ['linear', 'rbf']}
]
final_svm = tuneLearner("SVM", svm, svm_param_grid, feature_data, labels)
evaluateLearner("SVM", final_svm, feature_data, labels, 10)
