seed = 415
import time
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
from tensorflow.keras.utils import plot_model
# plot_model require pydot and graphviz
from keras_visualizer import visualizer

###################################
# running configurations
debugShortRun = False
crossValidationsGlobal = 10

# debugShortRun = True
# crossValidationsGlobal = 3

epochsGlobal = 10
batch_sizeGlobal = 64
startDir = 'C:/bgu/DSCI/DSCI'
# application expect data to be located at  :
#   <startDir>\data\profiles.dir\Plate_XYZK\profiles\mean_well_profiles.csv
#   <startDir>\data\chemical_annotations.csv
# experiments results will be written at :
#   <startDir>\ExperimentsResults
###################################

atomsPeriodicTable = ['c', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                      'Ar',
                      'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                      'Br', 'Kr', 'Rb',
                      'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                      'Xe', 'Cs',
                      'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                      'Hf', 'Ta',
                      'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                      'Th', 'Pa',
                      'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
                      'Hs', 'Mt',
                      'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

atomsPeriodicTableSortedGlobal = sorted(atomsPeriodicTable, key=len, reverse=True)
atomsCountMaxGlobal = [0 for i in range(len(atomsPeriodicTableSortedGlobal))]  # initialize with 0's
usedAtomsCountMaxGlobal = None
usedAtomsIndexesHashGlobal = {}
# look like {8: 3087, 104: 29879, 107: 30351, 108: 28458, 109: 29389, 112: 8421, 105: 19802, 110: 5751, 97: 1828, 11: 812, 116: 36, 24: 691, 111: 37, 38: 1}
modelGlobal = None
parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal = None
usedAtomsGlobal = None
RMSEActualGlobal = []
RMSERandomGlobal = []
countTreatedWellsGlobal = 0
countControlWellsGlobal = 0

chemical_annotationsFile = '/data/chemical_annotations.csv'
profilesDir = startDir + '/data/profiles.dir'

#############################
# this is a helper code / Print to File
import datetime
import os
import matplotlib.pyplot as plt

stringToPrintToFileGlobal = ""


def printDebug(strLog):
    global stringToPrintToFileGlobal
    initLogFileNameIfNeeded()
    dateString = genDateString()
    strLog = dateString + " " + strLog
    print(strLog)
    stringToPrintToFileGlobal = stringToPrintToFileGlobal + strLog + "\r"


def initLogFileNameIfNeeded():
    if (_FileName_ == ""):
        setLogFileName(str(genDateString()))


def genDateString():
    dateString = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return dateString


_FileName_ = ""


def setLogFileName(fileName):
    global _FileName_
    _FileName_ = fileName


def printToFile(fileName=""):
    if (fileName == ""):
        fileName = _FileName_
    fileName = fileNameToFullPath(fileName)
    global stringToPrintToFileGlobal
    file1 = open(fileName, "a")
    # file1.write("\r************************\r")
    file1.write(stringToPrintToFileGlobal)
    stringToPrintToFileGlobal = ""
    file1.close()


def fileNameToFullPath(fileName, ext=".log"):
    fileName = startDir + "/ExperimentsResults/" + fileName + ext
    return fileName


def renameToFinalLog(src, trgt):
    os.rename(fileNameToFullPath(src), fileNameToFullPath(trgt))


def plotToFile(fileName):
    plt.savefig(fileNameToFullPath(fileName, '.png'))


#############################

def dropNonUsedAtoms(parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes):
    # parsedChemicalAnnotationSmiles_AllAtoms looks like
    # [['BRD-A56675431-001-04-0', [0, 0, 0, ... 5, 3, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0]],[...]]]
    # usedAtomsIndexes looks like : [8, 104, 107, 108, 109, 112, 105, 110, 97, 11, 116, 24, 111, 38]
    usedAtoms = list(map(lambda i: atomsPeriodicTableSortedGlobal[i], usedAtomsIndexes))
    parsedChemicalAnnotationSmiles_usedAtoms_Hash = {}
    parsedChemicalAnnotationSmiles_usedAtoms = []
    for l in parsedChemicalAnnotationSmiles_AllAtoms:
        parsedChemicalAnnotationSmiles_usedAtomsForTreatment = list(map(lambda i: l[1][i], usedAtomsIndexes))
        parsedChemicalAnnotationSmiles_usedAtoms.append([l[0], parsedChemicalAnnotationSmiles_usedAtomsForTreatment])
        parsedChemicalAnnotationSmiles_usedAtoms_Hash[l[0]] = parsedChemicalAnnotationSmiles_usedAtomsForTreatment
    # printDebug(ret)
    usedAtomsCountMax = list(map(lambda i: atomsCountMaxGlobal[i], usedAtomsIndexes))
    return parsedChemicalAnnotationSmiles_usedAtoms, usedAtoms, usedAtomsCountMax, parsedChemicalAnnotationSmiles_usedAtoms_Hash


def parseChemicalAnnotationsFileSmiles(rawsChemicalAnnotationsFile):
    newRaws = []
    for raw in rawsChemicalAnnotationsFile:
        # each raw looks like: treatmentId , Smile
        # ['BRD-A56675431-001-04-0', 'NS(=O)(=O)c1cc2c(NC(CSCC=C)NS2(=O)=O)cc1Cl']
        atomsCount = parseSMILE(raw[1])
        newRaw = [raw[0]]
        newRaw.append(atomsCount)
        newRaws.append(newRaw)
        # newRaw looks like that:
        # ['BRD-A56675431-001-04-0', [0, 0, 0, ... 5, 3, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0]]
        # usedAtomsIndexesHash will contain all the indexes that were in use by all treatments in the plate
    return newRaws, list(usedAtomsIndexesHashGlobal.keys())


def readChemicalAnnotationsFile():
    # read chemical_annotations data
    #   there are errors in the data ,
    #   the chemical_annotations file contain more values per row then expected in some rows --> can't use numpy.genfromtxt or pd.read_csv
    raws = []
    with open(startDir + chemical_annotationsFile, 'r') as file:
        headerRaw = []
        reader = csv.reader(file)
        # printDebug(reader)
        for raw in reader:
            raw = [raw[0], raw[len(raw) - 2]]
            if (len(headerRaw) == 0):
                # col BROAD_ID, CPD_SMILES
                headerRaw = raw
            else:
                raws.append(raw)
    return raws


def parseSMILE(smile):
    global atomsPeriodicTableSortedGlobal, atomsCountMaxGlobal, usedAtomsIndexesHashGlobal
    # printDebug(smile)
    atomsCount = countAtomsInSmile(atomsPeriodicTableSortedGlobal, smile)

    # update global variables
    atomsCountMaxGlobal = np.maximum(atomsCountMaxGlobal, atomsCount).tolist()
    usedAtoms, usedAtomsIndexes = getUsedAtoms(atomsPeriodicTableSortedGlobal, atomsCount)
    for usedIndex in usedAtomsIndexes:
        if (usedIndex in usedAtomsIndexesHashGlobal):
            usedAtomsIndexesHashGlobal[usedIndex] = usedAtomsIndexesHashGlobal[usedIndex] + 1
        else:
            usedAtomsIndexesHashGlobal[usedIndex] = 1
    # printDebug(smile)
    # printDebug(atomsCount)
    # printDebug(atomsCountMax)
    # printDebug(usedAtoms)
    # printDebug(usedAtomsIndexes)
    return atomsCount


def getUsedAtoms(atoms, atomsCount):
    usedAtoms = []
    usedAtomsIndexes = []
    for i in range(len(atoms) - 1):
        if (atomsCount[i] > 0):
            usedAtoms.append(atoms[i])
            usedAtomsIndexes.append(i)
        else:
            continue
    return usedAtoms, usedAtomsIndexes


def countAtomsInSmile(atoms, smile):
    atomsCount = []
    for atom in atoms:
        atomCount = smile.count(atom)
        smile = smile.replace(atom, '')
        atomsCount.append(atomCount)
    return atomsCount


def readPlateData(mean_well_profilesFile):
    mean_well_profilesFileDF = pd.read_csv(mean_well_profilesFile)
    return mean_well_profilesFileDF


def splitControlAndTreated(mean_well_profilesFileDF):
    global parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal, countTreatedWellsGlobal, countControlWellsGlobal
    wellControl = mean_well_profilesFileDF.loc[mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    countControlWellsGlobal = countControlWellsGlobal + len(wellControl)
    wellTreatment = mean_well_profilesFileDF.loc[~mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    countTreatedWellsGlobal = countTreatedWellsGlobal + len(wellTreatment)
    # treatmentForLine = parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal[mean_well_profilesFileDF.iloc[0]['Metadata_pert_mfc_id']]
    return wellControl, wellTreatment


def normalizeTreatedWells(wellControl, wellTreatment):
    # Normalize treatment data using the control average
    # avg control data
    wellsDataStartColumnNumber = 17  # Cells_AreaShape_Area is the first numeric data column
    wellControlNumericData = wellControl.iloc[:, 17:]
    avgWellControlNumericData = wellControlNumericData.mean(axis=0)
    # reduce the avg from the actual data - to normalize
    normalizedWellTreatment = wellTreatment.apply(lambda x:
                                                  x[wellsDataStartColumnNumber:] - avgWellControlNumericData, axis=1)
    # set the treatment formula ID to the normalized treatments
    Metadata_pert_mfc_ids = wellTreatment.loc[:, 'Metadata_pert_mfc_id']
    return normalizedWellTreatment, Metadata_pert_mfc_ids


def selectTrainAndValidationPlates(crossValidationIdx, crossValidations):
    platesOnDisk = os.listdir(profilesDir)
    # random.shuffle(platesOnDisk) # We don' t shuffle in any stage ! we want the order to stay between the cross validaiton steps
    numOfPlatesToUse = len(platesOnDisk)
    if (debugShortRun):
        printDebug("### DEBUG !!! : run on only [" + str(crossValidations) + "]plates ###")
        numOfPlatesToUse = crossValidations  # todo: debug !!! to avoid reading the whole data while debugging!!!
    validationSize = (int)((1 / crossValidations) * numOfPlatesToUse)
    platesOnDiskNP = np.array(platesOnDisk)
    validationIdxs = list(range(validationSize * crossValidationIdx, validationSize * (crossValidationIdx + 1)))
    trainingIdxs = list(set(range(numOfPlatesToUse)) - set(validationIdxs))
    validationPlates = platesOnDiskNP[validationIdxs]
    trainingPlates = platesOnDiskNP[trainingIdxs]
    return trainingPlates, validationPlates


def preprocessTreatments():
    global usedAtomsCountMaxGlobal, parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal
    # read treatments formulas
    rawsChemicalAnnotationsFile = readChemicalAnnotationsFile()
    # parse  all treatments formulas
    parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes = \
        parseChemicalAnnotationsFileSmiles(rawsChemicalAnnotationsFile)
    # keep only used atoms in the smiles formula, atoms data and count statistics
    parsedChemicalAnnotationSmiles_usedAtoms, usedAtoms, usedAtomsCountMaxGlobal, parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal = \
        dropNonUsedAtoms(parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes)
    numOfTreatments = len(parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal)
    printDebug("numOfTreatments[" + str(numOfTreatments) + "]")
    # 30616
    printDebug("usedAtoms[" + str(usedAtoms) + "]")
    # ['Cl', 'c', 'C', 'N', 'O', 'S', 'H', 'F', 'Cn', 'Sc', 'I', 'Br', 'P', 'Sn']
    printDebug("numOfTreatmentsUsedTheAtom[" + str(usedAtomsIndexesHashGlobal.values()) + "]")
    # dict_values([3087, 29879, 30351, 28458, 29389, 8421, 19802, 5751, 1828, 812, 36, 691, 37, 1])
    printDebug("usedAtomsCountMax[" + str(usedAtomsCountMaxGlobal) + "]")
    # [5, 42, 62, 11, 19, 4, 24, 9, 2, 3, 6, 4, 2, 1]
    printDebug("total Treatments[" + str(len(parsedChemicalAnnotationSmiles_usedAtoms)) + "]")
    # '30616'

    # todo: calculate statistics about formulas (hystogram of used muleculas):
    return usedAtoms


def generateRandomTreatment():
    global usedAtomsCountMaxGlobal, parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal
    # generate smart random treatment
    smartRandom = []
    totalTreatments = len(parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal)
    usedAtomsNumList = list(usedAtomsIndexesHashGlobal.values())
    for index in range(len(usedAtomsIndexesHashGlobal)):
        usedRatio = usedAtomsNumList[index] / totalTreatments
        value = 0
        if random.random() <= usedRatio:
            maxVal = usedAtomsCountMaxGlobal[index]
            value = min((int)((random.random() * maxVal) + 1), maxVal)

        smartRandom.append(value)
    return smartRandom


def InitializeModelIfNeeeded(normalizedWellTreatment):
    global modelGlobal, usedAtomsGlobal
    # build model (wells to formula) # build ANN structure
    if (modelGlobal == None):
        modelGlobal = Sequential()
        input_dim = len(normalizedWellTreatment.columns)
        layers = [(int)(input_dim / 2), len(usedAtomsGlobal)]
        modelGlobal.add(Dense(layers[0], input_dim=input_dim, activation='sigmoid'))
        modelGlobal.add(Dense(layers[1], activation='sigmoid'))  # relu
        METRICS = [
            tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
        ]
        loss = tf.keras.losses.MeanSquaredError()  # todo: consider MeanAbsoluteError or MeanAbsolutePercentageError
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        # compile the keras model
        modelGlobal.compile(loss=loss, optimizer=optimizer, metrics=METRICS)


def fitModelWithPlateData(normalizedWellTreatment, treatmentsOfCurrentPlateDf):
    global modelGlobal, batch_sizeGlobal, epochsGlobal
    if (modelGlobal == None):
        printDebug('error! model is None')
        return
    fitBeginTime = time.time()
    # printDebug("start fit")
    modelGlobal.fit(x=normalizedWellTreatment,
                    y=treatmentsOfCurrentPlateDf,
                    batch_size=batch_sizeGlobal,
                    epochs=epochsGlobal,
                    use_multiprocessing=True,
                    verbose=2,
                    workers=3
                    # ,callbacks=[cp_callback]
                    )
    # printDebug("fit took[" + str(time.time() - fitBeginTime) + "]")


def validateModelWithPlate(plateNumber):
    global RMSERandomGlobal, RMSEActualGlobal
    print("validate with plate[" + str(plateNumber) + "]")
    Metadata_pert_mfc_ids, normalizedWellTreatment = preparePlateData(plateNumber)
    # run prediction
    prediction = modelGlobal.predict(normalizedWellTreatment)
    # get actual data of validation plates
    actualTreatmentsOfCurrentPlateDf = treatmentsIDsToData(Metadata_pert_mfc_ids)
    # calculate RMSE of prediction vs. actual validation plates data
    rmse = sqrt(mean_squared_error(prediction, actualTreatmentsOfCurrentPlateDf))
    print("RMSE[" + str(rmse) + "] actual")
    # generate a random prediction
    randomPrediction = list(map(lambda i: generateRandomTreatment(), actualTreatmentsOfCurrentPlateDf.values))
    # calculate RMSE of the random prediction vs. actual validation plates data
    rmseRandom = sqrt(mean_squared_error(randomPrediction, actualTreatmentsOfCurrentPlateDf))
    print("RMSE [" + str(rmseRandom) + "] Random")
    RMSEActualGlobal.append(rmse)
    RMSERandomGlobal.append(rmseRandom)
    # todo: Check if there is a diff of prediction between prediction on control and prediction done on treated plates


def trainModelWithPlate(plateNumber):
    print("train with plate[" + str(plateNumber) + "]")
    Metadata_pert_mfc_ids, normalizedWellTreatment = preparePlateData(plateNumber)
    treatmentsOfCurrentPlateDf = treatmentsIDsToData(Metadata_pert_mfc_ids)
    InitializeModelIfNeeeded(normalizedWellTreatment)
    fitModelWithPlateData(normalizedWellTreatment, treatmentsOfCurrentPlateDf)


def treatmentsIDsToData(Metadata_pert_mfc_ids):
    treatmentsOfCurrentPlate = list(
        map(lambda id: parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal[id], Metadata_pert_mfc_ids))
    treatmentsOfCurrentPlateDf = pd.DataFrame(treatmentsOfCurrentPlate)
    return treatmentsOfCurrentPlateDf


# Plate_25372
def preparePlateData(plateNumber):
    # printDebug("preparePlateData[" + plateNumber + "]")
    plateCsv = startDir + "/data/profiles.dir/" + plateNumber + "/profiles/mean_well_profiles.csv"
    # C:\bgu\DSCI\DSCI\data\profiles.dir\Plate_24279\profiles\mean_well_profiles.csv
    # read plates avg well data
    mean_well_profilesFileDF = readPlateData(plateCsv)
    # split control wells and treated wells
    wellControl, wellTreatment = splitControlAndTreated(mean_well_profilesFileDF)
    # Normalize treated wells with the plate control
    normalizedWellTreatment, Metadata_pert_mfc_ids = normalizeTreatedWells(wellControl, wellTreatment)
    # printDebug(normalizedWellTreatment)
    # printDebug(Metadata_pert_mfc_id)
    # todo: consider adding control wells with 0 treatment to the data
    return Metadata_pert_mfc_ids, normalizedWellTreatment


def plotModel(modelGlobal):
    plot_model(modelGlobal, to_file=startDir + "/ExperimentsResults/" + "model_plot.png",
               show_shapes=True,
               show_layer_names=True)
    #visualizer(modelGlobal, filename=startDir + "/ExperimentsResults/" + "model_plot2.png", format='png', view=True)


################################
def run():
    global modelGlobal, parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal, usedAtomsGlobal, crossValidationsGlobal, \
        countTreatedWellsGlobal, countControlWellsGlobal
    printDebug('start')
    startTime = time.time()
    printToFile()
    cur_dir = os.getcwd()
    printDebug("currentDir[" + str(cur_dir) + "]")
    os.path.isdir(cur_dir)

    # preprocess
    usedAtomsGlobal = preprocessTreatments()

    for crossValidationIdx in range(crossValidationsGlobal):
        modelGlobal = None  # initialize the model every x validation cycle
        # in  each iteraiton of the cross validation we prepare the whole plates data, and therefore those must be recount
        countTreatedWellsGlobal = 0
        countControlWellsGlobal = 0
        XValidaitonBeginTime = time.time()
        printDebug(
            "--- XValidaiton [" + str(crossValidationIdx) + "/" + str(crossValidationsGlobal) + "]-------------------")
        # select train and test plates from disk dirs
        trainingPlates, validationPlates = selectTrainAndValidationPlates(crossValidationIdx, crossValidationsGlobal)
        printDebug("Training plates[" + str(len(trainingPlates)) + "][" + str(trainingPlates) + "]")
        printDebug("Validation plates[" + str(len(validationPlates)) + "][" + str(validationPlates) + "]")
        for plateNumber in trainingPlates:
            trainModelWithPlate(plateNumber)

        printDebug("train took[" + str(time.time() - XValidaitonBeginTime) + "]")
        validaitonBeginTime = time.time()
        # run model predict on validation and calciulate RMSE and Random RMSE - per plate
        for plateNumber in validationPlates:
            validateModelWithPlate(plateNumber)

        printDebug(
            'RMSEActualGlobal mean[' + str(statistics.mean(RMSEActualGlobal)) + '][' + str(RMSEActualGlobal) + ']')
        printDebug(
            'RMSERandomGlobal mean[' + str(statistics.mean(RMSERandomGlobal)) + '][' + str(RMSERandomGlobal) + ']')
        printDebug("validaiton took[" + str(time.time() - validaitonBeginTime) + "]")
        printDebug(
            "XValidaiton (train + validaiton) took[" + str(time.time() - XValidaitonBeginTime)
            + "]validation[" + str(crossValidationIdx) + "/" + str(crossValidationsGlobal)
            + "]countTreatedWells[" + str(countTreatedWellsGlobal)
            + "]countControlWells[" + str(countControlWellsGlobal)
            + "]")
        printToFile()

    plotModel(modelGlobal)

    # output total rmse and Random rmse - for all plates in all cross validaitons
    printDebug('RMSEActualGlobal mean[' + str(statistics.mean(RMSEActualGlobal)) + '][' + str(RMSEActualGlobal) + ']')
    printDebug('RMSERandomGlobal mean[' + str(statistics.mean(RMSERandomGlobal)) + '][' + str(RMSERandomGlobal) + ']')
    printDebug("crossValidationsGlobal[" + str(crossValidationsGlobal)
               + "]epochsGlobal[" + str(epochsGlobal)
               + "]batch_sizeGlobal[" + str(batch_sizeGlobal) + "]")

    printDebug("run took[" + str(time.time() - startTime) + "]")
    printDebug('end ')
    printToFile()


run()
