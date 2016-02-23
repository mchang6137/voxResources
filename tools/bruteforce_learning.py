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
import datetime

import csv
import numpy as np
import itertools 

def evaluateLearner(name, learner, data, labels, splits):
    start = datetime.datetime.now()
    scores = cv.cross_val_score(learner, data, labels, cv = splits)
    end = datetime.datetime.now()
    print "-- %s Cross Validation Score --" % name
    print "Mean score: %f" % scores.mean()
    print "Standard deviation: %f" % scores.std()
    print "Elapsed time to calculate: " + str(end-start)
    return scores.mean()

fisher_directory = './../fisher_vectors/'
SONG_SET_SIZE = 1000
VALIDATION_SIZE = 20 * 10

max_score = -1
max_feature_set = []

#Choose a validation set
validation_list = [81, 1, 46, 71, 58, 63, 30, 44, 96, 77, 55, 98, 26, 49, 47, 85, 91, 53, 76, 39] #random.sample(range(0,99), 20)
print 'Our training set does not include the following: '
print validation_list

#Test Combinations of features
full_feature_list = ['FV_chroma.csv', 'FV_brightness.csv', 'FV_zerocross.csv', 'FV_eng.csv', 'FV_hcdf.csv', 'FV_roughness.csv', 'FV_t.csv', 'FV_MFCC.csv']
all_genre_feature = []

#Parse CSV file, where each row corresponds to a particular exempler
#Concatenate all the features in the same list
for feature_file in full_feature_list:
    feature_data = []
#    for x in range(0,SONG_SET_SIZE):
#        feature_data.append([])
    with open(fisher_directory + feature_file, 'rb') as f:
        reader = csv.reader(f)
        count = 0 
        for row in reader:
            #do not add the data if it is in the validation list
            if (count % 100) not in validation_list:
                float_row = [float(i) for i in row]
                feature_data.append(float_row)
            count += 1
        all_genre_feature.append(feature_data)

labels = [] 

#Labels will simply increment every 80
#Parse the labels file
with open('labels.csv', 'rb') as f:
    reader = csv.reader(f)
    for label in reader:
        count = 0
        for i in label:
            if count not in validation_list:
                labels.append(i)
            count += 1
            if count == 99:
                count = 0 

labels.append(10)
print len(labels)

labels = np.array(labels)

#Iterate through all combinations of features
for L in range(0, len(full_feature_list)+1):
    for subset in itertools.combinations(full_feature_list, L):
        feature_list = list(subset)
        if len(feature_list) == 0:
            continue

        print feature_list

        #Stack the Vectors
        feature_data = []
        for x in range(SONG_SET_SIZE - VALIDATION_SIZE):
            feature_data.append([])
 
        for feature in feature_list:
            feature_index = full_feature_list.index(feature)
            for songindex in range(len(feature_data)):
                feature_data[songindex] += (all_genre_feature[feature_index][songindex])
                    
        feature_data = np.array(feature_data)

        maxLearners = 100
        maxDepth = 5
        rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)

        mean_score = evaluateLearner("Random Forest", rf, feature_data, labels, 10)
        if mean_score > max_score:
            max_score = mean_score
            max_feature_set = feature_list
            
#evaluateLearner("Default AdaBoost", boost1, feature_data, labels, 10)
#evaluateLearner("AdaBoost with Rate 0.25", boost2, feature_data, labels, 10)
#evaluateLearner("AdaBoost with Rate 0.125", boost3, feature_data, labels, 10)

print 'max score is ' + str(max_score)
print 'max set is ' + str(max_feature_set)
