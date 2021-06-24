# Import required modules
import os
from shutil import copy2
import numpy as np
from sklearn.decomposition import PCA
import math
import csv
from matplotlib import pyplot as plt
import pandas as pd

# Function to copy the given files to the given folder
def copyFilesToPath(source, destination, files):
    for sample in files:
        copy2(source + sample, destination)

# This function splits the data into train and validation sets
# Seed is 1 in default
# First element inside of ratio is accepted as train set ratio and...
# ...second one is considered as validation set ratio
def trainValSplit(path, fileFormat, ratio, removeShort=False, threshold=10, seed=1):
    if(sum(list(ratio)) != 1):
        raise ValueError("Sum of the ratio elements should be equal to 1")
    # Set a random seed
    np.random.seed(seed)
    # Create a folder where the split files will be written
    if not os.path.exists(path + '/splitRes'):
        os.makedirs(path + '/splitRes')
    if not os.path.exists(path + '/splitRes/train'):
        os.makedirs(path + '/splitRes/train')
    if not os.path.exists(path + '/splitRes/validation'):
        os.makedirs(path + '/splitRes/validation')
    # Iterate over the users in a given folder
    for folder in os.listdir(path):
        # Make sure the folder name starts with 'user'
        if folder.startswith('user'):
            # Create a user folder inside of the output folders
            if not os.path.exists(path + '/splitRes/train/' + folder):
                os.makedirs(path + '/splitRes/train/' + folder)
            if not os.path.exists(path + '/splitRes/validation/' + folder):
                os.makedirs(path + '/splitRes/validation/' + folder)
            # Make sure that right data pieces are considered by checking their format
            samples = [elem for elem in os.listdir(path + '/' + folder) if elem.endswith(fileFormat)]
            # Remove the samples that is smaller than threshold
            longSamples = []
            if removeShort:
                for sample in samples:
                    label, array2D = readFromFile(path + '/' + folder + '/' + sample)
                    if len(array2D) > threshold:
                        longSamples.append(sample)
            # We give priority to training samples, thus we use ceil() function
            sizeOfTrain = int(math.ceil(len(longSamples) * list(ratio)[0]))
            trainSamples = np.random.choice(longSamples, sizeOfTrain, replace=False)
            # Remove those selected elements from the list which will give us validation set
            validationSamples = [i for i in longSamples if i not in trainSamples]
            # Add train samples to the relevant path
            copyFilesToPath(path + '/' + folder + '/', path + '/' + 'splitRes/' + 'train/' + folder, trainSamples)
            # Add validation samples to the relevant path
            copyFilesToPath(path + '/' + folder + '/', path + '/' + 'splitRes/' + 'validation/' + folder, validationSamples)

# Read the values from the lines of a given file
def readFromFile(filePath, delimiter = '\t'):
    # Initialize a feature vector and label
    array2D = []
    label = -1
    with open(filePath) as fileObject:
        # Skip the very first line which denotes the name of variables
        lines = csv.reader(fileObject, delimiter=delimiter)
        headers = next(lines)
        lines = list(lines)
        # If the second line is empty throw and exception
        try:
            # Remove the user tag from the label and convert it to int
            label = int(lines[0][0].replace('user', ''))
        except:
            print("Corrupted file detected, it will be deleted")
        for line in lines:
            array2D.append(list(map(int, line[1:3])))
    return label, array2D

# Remove the file in a given path
def removeFile(path):
    os.remove(path)

# This function reads files from the given path and deletes a...
# ...file which is empty or has few data points
def readDataset(path, fileFormat):
    # Initialize labels and features vector
    labels = []
    features = []
    # Iterate over folders
    for folder in os.listdir(path):
        # Make sure the folder name starts with 'user'
        if folder.startswith('user'):
            # Make sure that right data pieces are considered by checking their format
            samples = [elem for elem in os.listdir(path + '/' + folder) if elem.endswith(fileFormat)]
            for sample in samples:
                label, array2D = readFromFile(path + '/' + folder + '/' + sample)
                # Check if the file is in right format or not. i.e. if it returns -1...
                # ...then it is somehow corrupted, thus delete that file
                if(label == -1):
                    removeFile(path + '/' + folder + '/' + sample)
                else:
                    labels.append(label)
                    features.append(array2D)
    return labels, features

# Function to plot histogram of a list
def plotHistogram(data, numOfBins=20):
    # Set the parameters of bins
    bins = np.linspace(math.ceil(min(data)),
                       math.floor(max(data)),
                       numOfBins)

    plt.xlim([min(data) - 5, max(data) + 5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('Random Gaussian data (fixed number of bins)')
    plt.xlabel(f'variable ({numOfBins} evenly spaced bins)')
    plt.ylabel('count')
    # Display the histogram
    plt.show()

# Pad the sequence with given size
def padSequences(sequences, padSize):
    # Iterate over sequences and add [0, 0] of (padSize - len(elem)) size...
    # ...which is equal to the difference between biggest sample's size and...
    # ...current sample
    for index, elem in enumerate(sequences):
        sequences[index] = elem + [[0, 0]] * (padSize - len(elem))
    return sequences

def applyPCA(samples):
    data2D = []
    pca = PCA(n_components=1)
    for sample in samples:
        sample = [*zip(*sample)]
        for feature in sample:
            print(feature)
            pca.fit(list(feature))
            print(pca.singular_values_)

def standarize(data):
    array2D = []
    for sample in data:
            df = pd.DataFrame.from_records(sample)
            for column in df:
                df[column] = (df[column] - min(df[column]))/(max(df[column]) - min(df[column]))
            array2D.append(df.values.tolist())
    return array2D
