import matplotlib.pyplot as plt
import seaborn as sns;
import sys
import random
import scipy.io as spio
import numpy as np

import csv

#Plot the mean of the mfcc with each line colored by genre
#compute the mean for each song
genres = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

fisher_directory = './../fisher_vectors/'

feature_data = []
#Parse CSV file, wher each row corresponds to a particular exempler
with open(fisher_directory + 'FV_MFCC.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        float_row = np.array([float(i) for i in row])
        feature_data.append(float_row)        
feature_data = np.array(feature_data)

NUM_SONGS_PER_GENRE = 100

subplot_mean = [221, 222]
#subplot_stddev = [222, 224]
starting_index = 4

figure = plt.figure()
figure_count = 0 
for x in range(starting_index, starting_index + len(subplot_mean)):
    genre = 100 * x
    
    mean_list = []
    stddev_list = []

    for y in range(genre, genre + NUM_SONGS_PER_GENRE):
        mean_list.append(np.median(feature_data[y], axis = 0))
        stddev_list.append(np.std(feature_data[y], axis = 0))
 
    ax1 = figure.add_subplot(subplot_mean[figure_count])   
    ax1.hist(mean_list)
    plt.title("Fisher Vector MFCC Median for " + genres[x], fontsize=8)
    plt.xlabel("Median of MFCC", fontsize = 8)
    plt.ylabel("Frequency", fontsize=8)

    #ax2 = figure.add_subplot(subplot_stddev[figure_count])
    #ax2.hist(stddev_list)
    #plt.title("Standard deviation for " + genres[x], fontsize=8)
    #plt.xlabel("Std. dev of MFCC", fontsize=8)
    #plt.ylabel("Frequency", fontsize=8)
    
    figure_count += 1

plt.show()
    
    
    
