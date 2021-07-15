# Shahriyar Mammadli
# Master module to orchestrate the sub-modules for intrusion detection system.
# Import required libraries.
import os
import dataPreparation as dp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import helperFunctions as hf
import pandas as pd

# Set parameters.
# Pandas DataFrame printing options.
pd.options.display.max_columns = None
# A dictionary to hold features and their paths.
featureParameters = [{
    'name': 'mouse',
    'path': os.path.abspath('./data/MouseActions'),
    'cols': [1, 2, 3, 4, 5],
    'threshold': 15
    }, {
    'name': 'keyboard',
    'path': os.path.abspath('./data/Keystrokes'),
    'cols': [1, 2],
    'threshold': 20
    }
]
# Retrieve the longest sample size (time-steps) for mouse data.
longestMouseSample = dp.findLongestSample(os.path.join(featureParameters[0]['path'], 'genuine'))
# Retrieve the longest sample size (time-steps) for keyboard data.
longestKeyboardSample = dp.findLongestSample(os.path.join(featureParameters[1]['path'], 'genuine'))
# Set training parameters.
verbose, epochs, batchSize = 1, 8, 32


# A function to implement the experiment steps.
def runExperiment(featureObj, longestSampleLength,  repeats=10):
    print("Training a model on the feature " + featureObj['name'] + ".")
    # Processing and Loading the data.
    features, classes = dp.loadData(featureObj, longestSampleLength)
    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.2, random_state=44)
    # One-hot encode the classes.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # Repeat the experiment.
    scores = list()
    for r in range(repeats):
        if featureObj['name'] == 'keyboard':
            score = hf.keyboardModel(X_train, y_train, X_test, y_test, verbose, epochs, batchSize)
        elif featureObj['name'] == 'mouse':
            score = hf.mouseModel(X_train, y_train, X_test, y_test, verbose, epochs, batchSize)
        else:
            raise ValueError(f"The feature name {featureObj['name']} is not known.")
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # Summarize the results.
    hf.summarizeResults(scores)


# Run the experiment for mouse data.
# runExperiment(featureParameters[0], longestMouseSample)
# Run the experiment for keyboard data.
runExperiment(featureParameters[1], longestKeyboardSample)
