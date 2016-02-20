import sys
import random
import scipy.io as spio
import numpy as np

import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss

import csv
import numpy as np

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
            print count
            count += 1

print feature_data[0]

feature_data = np.array(feature_data)

maxLearners = 100
maxDepth = 5
N = len(feature_data)
labels = np.array([N,])

#Parse the labels file
with open('labels.csv', 'rb') as f:
    reader = csv.reader(f)
    for label in reader:
        labels = np.array([int(i) for i in label])

TEidx = np.array(random.sample(range(0,N), N/10))
X_TE = feature_data[TEidx, :]
X_TR = feature_data[[i for i in range(0,N) if i not in TEidx],:]
Y_TE = labels[TEidx]
Y_TR = labels[[i for i in range(0,N) if i not in TEidx]]

rf = RandomForestClassifier(n_estimators = maxLearners, max_depth = maxDepth, warm_start = False)
rf.fit(X_TR,Y_TR)
predictionsRF = rf.predict(X_TE)
errorRF = zero_one_loss(predictionsRF, Y_TE)
print errorRF



