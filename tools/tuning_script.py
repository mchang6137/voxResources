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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
import sklearn.cross_validation as cv

import csv
import numpy as np

# does grid search with the given learner, data, and labels on the
# specified param grid, prints best params, returns the best learner 
# produced by the search
def tuneLearner(name, learner, param_grid, data, labels, splits):
    search = GridSearchCV(estimator = learner, param_grid = param_grid, cv = splits)
    search.fit(data, labels)
    print "-- %s Grid Search Parameters --" % name
    print search.grid_scores_
    for param in search.best_params_:
        print "%s: %s" % (param, str(search.best_params_[param]))
    print "--\n"
    return search.best_estimator_

def evaluateLearner(name, learner, data, labels):
    accuracy = learner.score(data, labels)
    predictions = learner.predict(data)
    cm = confusion_matrix(labels, predictions)
    print "-- %s Results --" % name
    print "Mean accuracy: %f" % accuracy
    print cm
    print "--\n"

feature_list = ['FV_zerocross.csv', 'FV_eng.csv', 'FV_hcdf.csv', 'FV_MFCC.csv']
fisher_directory = './../fisher_vectors/'
feature_data = []
validation_data = []
SONG_SET_SIZE = 800
VALIDATION_SET_SIZE = 200

validation_indices = [1, 26, 30, 39, 44, 46, 47, 49, 53, 55, 58, 63, 71, 76, 77, 81, 85, 91, 96, 98]

for i in range(0,SONG_SET_SIZE):
    feature_data.append([])
for i in range(0,VALIDATION_SET_SIZE):
    validation_data.append([])

#Parse CSV file, where each row corresponds to a particular exempler
#Concatenate all the features in the same list
for feature_file in feature_list:
    with open(fisher_directory + feature_file, 'rb') as f:
        reader = csv.reader(f)
        temp_feature_data = []
        temp_validation_data = []

        i = 0
        for row in reader:
            data_row = [float(j) for j in row]
            if (i % 100) in validation_indices:
                temp_validation_data.append(data_row)
            else:
                temp_feature_data.append(data_row)
            i += 1

        i = 0
        for song_temp_feature in temp_feature_data:
            feature_data[i] += song_temp_feature
            i += 1
        i = 0
        for val_temp_feature in temp_validation_data:
            validation_data[i] += val_temp_feature
            i += 1

feature_data = np.array(feature_data)
validation_data = np.array(validation_data)

N = len(feature_data)
V = len(validation_data)
data_labels = np.array([N,])
val_labels = np.array([V,])

#Parse the labels file
with open(fisher_directory + 'labels.csv', 'rb') as f:
    reader = csv.reader(f)
    for label_list in reader:
        i = 0
        val_labels_temp = []
        data_labels_temp = []
        for label in label_list:
            if (i % 100) in validation_indices:
                val_labels_temp.append(int(label))
            else:
                data_labels_temp.append(int(label))
            i += 1
        val_labels = np.array(val_labels_temp)
        data_labels = np.array(data_labels_temp)

# tune random forest using grid search and evaluate
rf = RandomForestClassifier(warm_start = False)
rf_param_grid = [
    {'n_estimators': [10, 25, 50, 75, 100, 125, 150, 175], 'max_depth': [4, 5, 6, 7, 8, 9, 10]}
]
final_rf = tuneLearner("Random Forest", rf, rf_param_grid, feature_data, data_labels, 5)
final_rf.fit(feature_data, data_labels)
evaluateLearner("Random Forest", final_rf, validation_data, val_labels)

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
    {'C': [0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75], 'kernel': ['linear', 'rbf']}
]
final_svm = tuneLearner("SVM", svm, svm_param_grid, feature_data, data_labels, 5)
final_svm.fit(feature_data, data_labels)
evaluateLearner("SVM", final_svm, validation_data, val_labels)
