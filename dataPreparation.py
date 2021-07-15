# Shahriyar Mammadli
# Data preparation module for the system.
# Import required libraries.
import os
from collections import Counter
import pandas as pd
import numpy as np


# Load a sample from .txt file.
def loadSample(path):
    return pd.read_csv(path, delimiter="\t")


# Find the longest sample in terms of time-steps.
def findLongestSample(path):
    # Initialize a variable to hold the longest sample.
    longest = 0
    for sample in os.listdir(path):
        # Check if the file type is correct.
        if sample.endswith('.txt'):
            sampleDf = loadSample(os.path.join(path, sample)).iloc[:, 1:6]
            # Compare the maxSize and current sample's size and take the bigger one.
            longest = max(longest, len(sampleDf))
    return longest


# The function to process and return the data. It takes path to the folder...
# ...as a parameter. Also, timeThresh is a threshold to eliminate samples that are shorter...
# ...it (in terms of time-steps). The maxSize parameter is a necessary for padding step...
# ...since samples with shorter time-steps are needed to be padded to the size of longest. This...
# ...parameter is needed to be known in advance.
def loadData(featureObj, maxSize):
    # Initialize lists to hold features and their classes (i.o.w. predictors...
    # ...and their corresponding target values).
    features = list()
    classes = list()
    for sample in os.listdir(os.path.join(featureObj['path'], 'genuine')):
        # Check if the file type is correct.
        if sample.endswith('.txt'):
            # Load the sample into a DataFrame object.
            sampleDf = loadSample(os.path.join(featureObj['path'], 'genuine', sample)).iloc[:, featureObj['cols']]
            # Check if the sample length meets the threshold.
            if len(sampleDf) >= featureObj['threshold']:
                # Do feature engineering corresponding to the type of feature.
                if featureObj['name'] == 'keyboard':
                    # Min-normalize the features.
                    sampleDf = (minNormalization(sampleDf))
                    # Create a new column which contains the time between two key click interactions.
                    sampleDf['TimeBetweenKeyDowns'] = sampleDf['KeyDownTimeStamp'] - sampleDf['KeyDownTimeStamp'].shift(1, fill_value=0)
                    # Create a new column which contains the time between two key release interactions.
                    sampleDf['TimeBetweenKeyReleases'] = sampleDf['KeyReleaseTimeStamp'] - sampleDf['KeyReleaseTimeStamp'].shift(1, fill_value=0)
                    # Create a new column which contains the duration of time between two consecutive interactions.
                    sampleDf['ActionDuration'] = sampleDf['KeyReleaseTimeStamp'] - sampleDf['KeyDownTimeStamp']
                # Pad the sample to the size of longest sample.
                sampleDf = padSample(sampleDf, maxSize)
                # Add sample to the list.
                features.append(sampleDf.values)
                # Retrieve the class from the sample name and add it to the classes list.
                classes.append(int(sample.split('_')[0].split('user')[1]))
    # Stack the samples in 3-D format of samples, time-steps, features.
    features = np.stack(features)
    # Force classes to have consecutive labels starting from 0. I.o.w. there should not be any missing label in...
    # ... series. E.g. 1, 2, 4, 7 is converted to 0, 1, 2, 3, 4
    classes = forceConsecutive(classes)
    return features, np.asarray(classes, dtype=np.float32)


# The function to min-normalize a sample. Since the very first sample is the smallest (minimum) value...
# ...in the DataFrame subtract that from all the samples.
def minNormalization(df):
    # Subtract the smallest value (at the very first cell, since it is time-series data) from all cells.
    return df - df.iloc[0, 0]


# The function to pad a sample to the size of longest sample.
def padSample(orgDf, maxSize):
    # Create a DataFrame object with the vales of 0. Shape of the object is nrows = maxSize-len(orgDf)...
    # ...and ncols = same as orgDf.
    padDf = pd.DataFrame(0, index=np.arange(maxSize-len(orgDf)), columns=orgDf.columns)
    # Concatenate the DataFrame objects.
    mergedDf = pd.concat([orgDf, padDf], ignore_index=True, sort=False)
    return mergedDf


# A function to find the missing integers in a given range.
def missingElements(classes):
    start, end = 0, max(classes)
    return sorted(set(range(start, end + 1)).difference(classes))


# A function to force the list of integers to be consecutive integers starting from the zero. In other...
# ...words, make sure that there is no any missing integer in the list.
def forceConsecutive(classes):
    # Find the missing integers in the list.
    missing = missingElements(classes)
    # For each element in the list remove the number of missing elements that is smaller that the element.
    classes = [j - sum(i < j for i in missing) for j in classes]
    return classes
