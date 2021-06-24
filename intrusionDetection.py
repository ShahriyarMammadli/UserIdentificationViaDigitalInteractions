# Import required models
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
import pickle
import dataPreparation as dp
import helperFunctions as hf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Initialize the parameters
# Seed value for randomization
seed = 2020
# Initialize train validation split ratio
ratio = (0.5, 0.5)
# We set threshold to 10. The samples that are shorter than 10 will be removed
threshold = 10
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 100, 10, 8
    n_timesteps, n_features, n_outputs = 852, 2, 25
    trainX = np.asarray(trainX)
    testX = np.asarray(testX)
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(featuresTrain, labelsTrain, featuresVal, labelsVal, repeats=10):
    import time
    start = time.time()
    # load data
    trainX, trainy, testX, testy = featuresTrain, labelsTrain, featuresVal, labelsVal
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    end = time.time()
    elapsed = end - start
    print(f"Training took {elapsed} seconds")
    # summarize results
    start = time.time()
    summarize_results(scores)
    end = time.time()
    elapsed = end - start
    print(f"Testing took {elapsed} seconds")

# Perform the data preparation phase
# Choose whether do the train and validation split. Default value...
# ...is False. If it is first time run, make it True. Decomment...
# ...the following line for this purpose.
dp.prepareData(ratio, trainValSplit=True, threshold=threshold, seed=seed)

with open('trainData.pickle', 'rb') as handle:
    labelsTrain, featuresTrain, labelsVal, featuresVal = pickle.load(handle)

# Some of the samples includes very few data points of time series...
# ...which needs to extracted. To achieve that we need to firstly...
# ...specify a threshold. Samples that have less data points than...
# ...threshold will be removed. For this purpose we are using histogram...
# ...to analyze the distribution and frequency of lengths.
# Other than that, we need to specify pad length, to which length, we...
# ...will pad the samples. To achieve that, we will use the maximum length in...
# ...the training set.
# In first run we set threshold to 0, then after plotting the histogram...
# ...and choose a value for threshold parameter, then set it for next runs.
# Plot the histogram of the samples' length
# hf.plotHistogram([len(i) for i in featuresTrain + featuresVal], numOfBins=300)

# By the help of histogram we choose the threshold as 10
# Print the lengths of maximum samples of train set and validation set
print(f'The maximum length among the samples is {max([len(i) for i in featuresTrain])} in TRAINING set')
print(f'The maximum length among the samples is {max([len(i) for i in featuresVal])} in VALIDATION set')
# Max value for Training set is 852, and 745 for Validation set which will make our work easier...
# ...since if any of the samples in Validation set would be bigger than the maximum value of...
# ...the training set we would need to reduce their size.
padSize = max([len(i) for i in featuresTrain])
# Pad sequences
featuresTrain = hf.padSequences(featuresTrain, padSize)
featuresVal = hf.padSequences(featuresVal, padSize)

run_experiment(featuresTrain, labelsTrain, featuresVal, labelsVal)