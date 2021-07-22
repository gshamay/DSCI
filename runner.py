seed = 415
import time
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

from io import StringIO
from sklearn.metrics import roc_auc_score
from scipy import stats

debugShortRun = False
debugShortRun = True

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

startDir = 'C:/bgu/DSCI/DSCI'
chemical_annotationsFile = '/data/chemical_annotations.csv'
mean_well_profilesFile = '/data/mean_well_profiles.csv'
profilesDir = startDir + '/data/profiles.dir'


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
    # print(ret)
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
        # print(reader)
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
    # print(smile)
    atomsCount = countAtomsInSmile(atomsPeriodicTableSortedGlobal, smile)

    # update global variables
    atomsCountMaxGlobal = np.maximum(atomsCountMaxGlobal, atomsCount).tolist()
    usedAtoms, usedAtomsIndexes = getUsedAtoms(atomsPeriodicTableSortedGlobal, atomsCount)
    for usedIndex in usedAtomsIndexes:
        if (usedIndex in usedAtomsIndexesHashGlobal):
            usedAtomsIndexesHashGlobal[usedIndex] = usedAtomsIndexesHashGlobal[usedIndex] + 1
        else:
            usedAtomsIndexesHashGlobal[usedIndex] = 1
    # print(smile)
    # print(atomsCount)
    # print(atomsCountMax)
    # print(usedAtoms)
    # print(usedAtomsIndexes)
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
    global parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal
    wellControl = mean_well_profilesFileDF.loc[mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    wellTreatment = mean_well_profilesFileDF.loc[~mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    treatmentForLine = parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal[
        mean_well_profilesFileDF.iloc[0]['Metadata_pert_mfc_id']]
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
        print("### DEBUG !!! : run on only [" + str(crossValidations) + "]plates ###")
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
    print("numOfTreatments[" + str(numOfTreatments) + "]")
    # 30616
    print("usedAtoms[" + str(usedAtoms) + "]")
    # ['Cl', 'c', 'C', 'N', 'O', 'S', 'H', 'F', 'Cn', 'Sc', 'I', 'Br', 'P', 'Sn']
    print("numOfTreatmentsUsedTheAtom[" + str(usedAtomsIndexesHashGlobal.values()) + "]")
    # dict_values([3087, 29879, 30351, 28458, 29389, 8421, 19802, 5751, 1828, 812, 36, 691, 37, 1])
    print("usedAtomsCountMax[" + str(usedAtomsCountMaxGlobal) + "]")
    # [5, 42, 62, 11, 19, 4, 24, 9, 2, 3, 6, 4, 2, 1]
    print("total Treatments[" + str(len(parsedChemicalAnnotationSmiles_usedAtoms)) + "]")
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
            value = (int)(random.random() * usedAtomsCountMaxGlobal[index])

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
    global modelGlobal
    if (modelGlobal == None):
        print('error! model is None')
        return
    fitBeginTime = time.time()
    print("start fit")
    epochs = 10
    batch_size = 64
    modelGlobal.fit(x=normalizedWellTreatment,
                    y=treatmentsOfCurrentPlateDf,
                    batch_size=batch_size,
                    epochs=epochs,
                    use_multiprocessing=True,
                    verbose=2,
                    workers=3
                    # ,callbacks=[cp_callback]
                    )
    print("fit took[" + str(time.time() - fitBeginTime) + "]")


def validateModelWithPlate(plateNumber):
    global RMSERandomGlobal, RMSEActualGlobal
    print("validate plate[" + str(plateNumber) + "]")
    Metadata_pert_mfc_ids, normalizedWellTreatment = preparePlateData(plateNumber)
    prediction = modelGlobal.predict(normalizedWellTreatment)

    # todo: calculate RMSE for validation plate
    actualTreatmentsOfCurrentPlateDf = treatmentsIDsToData(Metadata_pert_mfc_ids)
    rmse = sqrt(mean_squared_error(prediction, actualTreatmentsOfCurrentPlateDf))
    print("RMSE[" + str(rmse) + "] actual")
    randomPrediction = list(map(lambda i: generateRandomTreatment(), actualTreatmentsOfCurrentPlateDf.values))
    rmseRandom = sqrt(mean_squared_error(randomPrediction, actualTreatmentsOfCurrentPlateDf))
    print("RMSE [" + str(rmseRandom) + "] Random")
    RMSEActualGlobal.append(rmse)
    RMSERandomGlobal.append(rmseRandom)
    # todo: Check diff of prediction between prediction on control and prediction done on treated plates


def trainModelWithPlate(plateNumber):
    Metadata_pert_mfc_ids, normalizedWellTreatment = preparePlateData(plateNumber)
    treatmentsOfCurrentPlateDf = treatmentsIDsToData(Metadata_pert_mfc_ids)
    InitializeModelIfNeeeded(normalizedWellTreatment)
    fitModelWithPlateData(normalizedWellTreatment, treatmentsOfCurrentPlateDf)


def treatmentsIDsToData(Metadata_pert_mfc_ids):
    treatmentsOfCurrentPlate = list(
        map(lambda id: parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal[id], Metadata_pert_mfc_ids))
    treatmentsOfCurrentPlateDf = pd.DataFrame(treatmentsOfCurrentPlate)
    return treatmentsOfCurrentPlateDf


def preparePlateData(plateNumber):
    print("plate[" + plateNumber + "]")
    plateCsv = startDir + "/data/profiles.dir/" + plateNumber + "/profiles/mean_well_profiles.csv"
    # C:\bgu\DSCI\DSCI\data\profiles.dir\Plate_24279\profiles\mean_well_profiles.csv
    # read plates avg well data
    mean_well_profilesFileDF = readPlateData(plateCsv)
    # split control wells and treated wells
    wellControl, wellTreatment = splitControlAndTreated(mean_well_profilesFileDF)
    # Normalize treated wells with the plate control
    normalizedWellTreatment, Metadata_pert_mfc_ids = normalizeTreatedWells(wellControl, wellTreatment)
    # print(normalizedWellTreatment)
    # print(Metadata_pert_mfc_id)
    # todo: add control wells with 0 treatment to the data
    return Metadata_pert_mfc_ids, normalizedWellTreatment


################################
def run():
    global modelGlobal, parsedChemicalAnnotationSmiles_usedAtoms_HashGlobal, usedAtomsGlobal
    print('start')
    cur_dir = os.getcwd()
    print("currentDir[" + str(cur_dir) + "]")
    os.path.isdir(cur_dir)

    # preprocess
    usedAtomsGlobal = preprocessTreatments()

    crossValidations = 3
    for crossValidationIdx in range(crossValidations):
        print("--- XValidaiton [" + str(crossValidationIdx) + "/" + str(crossValidations) + "]-------------------")
        # select train and test plates from disk dirs
        trainingPlates, validationPlates = selectTrainAndValidationPlates(crossValidationIdx, crossValidations)
        for plateNumber in trainingPlates:
            trainModelWithPlate(plateNumber)

        # run model predict on validation
        for plateNumber in validationPlates:
            validateModelWithPlate(plateNumber)


    # output total rmse and rmse diff

    print('RMSEActualGlobal mean[' + str(statistics.mean(RMSEActualGlobal)) + '][' + str(RMSEActualGlobal) + ']')
    print('RMSERandomGlobal mean[' + str(statistics.mean(RMSERandomGlobal)) + '][' + str(RMSERandomGlobal) + ']')
    print('end')


run()
