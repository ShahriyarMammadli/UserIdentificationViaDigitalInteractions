# Shahriyar Mammadli
# Module which is comprised of the helper functions for intrusion detection system.
# Import required libraries.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split


# The function that builds 1-D CNN  with specified parameters and given train-test data.
def keyboardModel(trainX, trainy, testX, testy, verbose, epochs, batchSize):
    # Obtain the sizes of timeSteps, features and outputs from the data.
    timeSteps, features, outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # Form the CNN structure.
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=14, activation='relu', input_shape=(timeSteps, features)))
    model.add(Conv1D(filters=32, kernel_size=10, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the network.
    model.fit(trainX, trainy, epochs=epochs, batch_size=batchSize, verbose=verbose)
    # Evaluate the model.
    _, trainAccuracy = model.evaluate(trainX, trainy, batch_size=batchSize, verbose=0)
    _, testAccuracy = model.evaluate(testX, testy, batch_size=batchSize, verbose=0)
    print(trainAccuracy, testAccuracy)
    return testAccuracy


# The function that builds 1-D CNN  with specified parameters and given train-test data.
def mouseModel(trainX, trainy, testX, testy, verbose, epochs, batchSize):
    # Obtain the sizes of timeSteps, features and outputs from the data.
    timeSteps, features, outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    trainX, valX, trainy, valY = train_test_split(trainX, trainy, test_size=0.25, random_state=1)
    # Form the CNN structure.
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=12, activation='relu', input_shape=(timeSteps, features)))
    model.add(Conv1D(filters=32, kernel_size=12, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the network.
    model.fit(trainX, trainy, validation_data=(), epochs=epochs, batch_size=batchSize, verbose=verbose)
    # Evaluate the model.
    _, trainAccuracy = model.evaluate(trainX, trainy, batch_size=batchSize, verbose=0)
    _, testAccuracy = model.evaluate(testX, testy, batch_size=batchSize, verbose=0)
    print(trainAccuracy, testAccuracy)
    return testAccuracy


# A function to summarize the run results.
def summarizeResults(scores):
    # Print the accuracies of all runs and then the maximum accuracy.
    print(scores)
    print('Maximum accuracy: %.3f%%' % (max(scores)))
