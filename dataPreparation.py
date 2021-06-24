# Import required modules
import helperFunctions as hf
import pickle
# Set dataset root path
rootPath = "C:/Users/smammadli/Desktop/Store/DataSets/Raw/User-Digital-Print/ISOT_Web_Interactions(Mouse_Keystroke_SiteAction)_Dataset"
# Set file format of the data
fileFormat = '.txt'
# Split the keystroke data into train and validation
# If train validation
def prepareData(ratio, trainValSplit = False, threshold=10, seed=1):
    if trainValSplit:
        hf.trainValSplit(rootPath + "/Keystrokes/genuine", fileFormat, ratio=ratio, removeShort=True, threshold=threshold, seed=seed)
        # hf.trainValSplit(rootPath + "/MouseActions/genuine", fileFormat, ratio=ratio, removeShort=True, threshold=threshold, seed=seed)
        # hf.trainValSplit(rootPath + "/SiteActions/genuine", fileFormat, ratio=ratio, removeShort=True, threshold=threshold, seed=seed)

    # Read the data set into a high-dimensional space
    labelsTrain, featuresTrain = hf.readDataset(rootPath + "/Keystrokes/genuine/splitRes/train", fileFormat)
    # Read the data set into a high-dimensional space
    labelsVal, featuresVal = hf.readDataset(rootPath + "/Keystrokes/genuine/splitRes/validation", fileFormat)
    # Write the processed data into a pickle file
    with open('trainData.pickle', 'wb') as handle:
        pickle.dump([labelsTrain, featuresTrain, labelsVal, featuresVal], handle, protocol=pickle.HIGHEST_PROTOCOL)
