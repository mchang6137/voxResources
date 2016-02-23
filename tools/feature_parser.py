import numpy as np
import matplotlib.pyplot as plt
import datetime

dir = './../fisher_vectors/'
filename = 'new_genre_data_with_validation.txt'

feature_list = []

#Finds the nth highest scores
with open(dir + filename) as infile:
    #Read the lines four a time
    while True:
        feature_set = infile.readline() 
        tagline = infile.readline()
        mean_score = infile.readline()
        standard_deviation = infile.readline()
        elapsed_time = infile.readline()
        if mean_score == '':
            break
        float_mean = float(mean_score.split(' ')[2])
        tuple = (float_mean, feature_set, elapsed_time)
        feature_list.append(tuple)
        
sorted_vector = sorted(feature_list, key=lambda x: x[0])
print sorted_vector
#Analyze the feature set
feature_dict = {}

index = 0
exclude_MFCC = False
#Remove all feature sets without the feature MFCC
if exclude_MFCC is True:
    while index < len(sorted_vector):
        (score, features, elapsed_time) = sorted_vector[index]
        feature_quotes = features.split('\'')[1:]
        all_features = feature_quotes[0::2]
    
        mfcc_present = False
        for feature in all_features:
            if feature == 'FV_MFCC.csv':
                mfcc_present = True
                break
    
        if mfcc_present is True:
            del sorted_vector[index]
        else:
            index += 1
    
print sorted_vector
    
entries_parsed = 0
mean_sum = 0
for index in range(len(sorted_vector)-255, len(sorted_vector)):
    entries_parsed +=1 
    (score, features, elapsed_time) = sorted_vector[index]
    print float(score)
    mean_sum += float(score)
    feature_quotes = features.split('\'')[1:]
    all_features = feature_quotes[0::2]

    for feature in all_features:
        if feature in feature_dict:
            feature_dict[feature] += 1
        else:
            feature_dict[feature] = 1

print 'mean is ' + str(mean_sum / entries_parsed)
print feature_dict
        
#Measure runtime
#Average across all combinations with that feature set
full_feature_list = ['FV_chroma.csv', 'FV_brightness.csv', 'FV_zerocross.csv', 'FV_eng.csv', 'FV_hcdf.csv', 'FV_roughness.csv', 'FV_t.csv', 'FV_MFCC.csv']
feature_time = {}

for index in range(len(sorted_vector)-255, len(sorted_vector)):
    (score, features, elapsed_time) = sorted_vector[index]
    elapsed_time = float(elapsed_time.split(':')[3])
    feature_quotes = features.split('\'')[1:]
    all_features = feature_quotes[0::2]
    

    for feature in all_features:
        if feature in feature_time:
            feature_time[feature].append(elapsed_time)
        else:
            feature_time[feature] = []
            feature_time[feature].append(elapsed_time)

for feature in feature_time:
    print feature
    print 'mean time ' + str(np.mean(feature_time[feature]))





