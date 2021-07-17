seed = 415
import time
import csv
import pandas as pd
import numpy as np
import zipfile
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

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

atomsPeriodicTableSorted = sorted(atomsPeriodicTable, key=len, reverse=True)
atomsCountMax = [0 for i in range(len(atomsPeriodicTableSorted))]
usedAtomsIndexesHash = {}
chemical_annotationsFile = './data/chemical_annotations.csv'
mean_well_profilesFile = './data/mean_well_profiles.csv'


def dropNonUsedAtoms(parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes):
    usedAtoms = list(map(lambda i: atomsPeriodicTableSorted[i], usedAtomsIndexes))

    parsedChemicalAnnotationSmiles_usedAtoms_Hash = {}
    parsedChemicalAnnotationSmiles_usedAtoms = []
    for l in parsedChemicalAnnotationSmiles_AllAtoms:
        parsedChemicalAnnotationSmiles_usedAtomsForTreatment = list(map(lambda i: l[1][i], usedAtomsIndexes))
        parsedChemicalAnnotationSmiles_usedAtoms.append([l[0], parsedChemicalAnnotationSmiles_usedAtomsForTreatment])
        parsedChemicalAnnotationSmiles_usedAtoms_Hash[l[0]] = parsedChemicalAnnotationSmiles_usedAtomsForTreatment
    # print(ret)
    usedAtomsCountMax = list(map(lambda i: atomsCountMax[i], usedAtomsIndexes))
    return parsedChemicalAnnotationSmiles_usedAtoms, usedAtoms, usedAtomsCountMax, parsedChemicalAnnotationSmiles_usedAtoms_Hash


def parseChemicalAnnotationsFileSmiles(rawsChemicalAnnotationsFile):
    newRaws = []
    for raw in rawsChemicalAnnotationsFile:
        atomsCount = parseSMILE(raw[1])
        newRaw = [raw[0]]
        newRaw.append(atomsCount)
        newRaws.append(newRaw)
    return newRaws, list(usedAtomsIndexesHash.values())


def readChemicalAnnotationsFile():
    # read chemical_annotations data
    #   there are erros in the data ,
    #   the chemical_annotations file contain more values per row then expected in some rows --> can't use numpy.genfromtxt or pd.read_csv
    raws = []
    with open(chemical_annotationsFile, 'r') as file:
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
    global atomsPeriodicTableSorted, atomsCountMax, usedAtomsIndexesHash
    # print(smile)
    atomsCount = countAtomsInSmile(atomsPeriodicTableSorted, smile)

    # update global variables
    atomsCountMax = np.maximum(atomsCountMax, atomsCount).tolist()
    usedAtoms, usedAtomsIndexes = getUsedAtoms(atomsPeriodicTableSorted, atomsCount)
    for usedIndex in usedAtomsIndexes:
        usedAtomsIndexesHash[usedIndex] = usedIndex

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


def run():
    print('start')
    # read treatments formulas
    rawsChemicalAnnotationsFile = readChemicalAnnotationsFile()
    # parse  all treatments formulas
    parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes = \
        parseChemicalAnnotationsFileSmiles(rawsChemicalAnnotationsFile)

    # keep only used atoms in the smiles formula, atoms data and count statistics
    parsedChemicalAnnotationSmiles_usedAtoms, usedAtoms, usedAtomsCountMax, parsedChemicalAnnotationSmiles_usedAtoms_Hash = \
        dropNonUsedAtoms(parsedChemicalAnnotationSmiles_AllAtoms, usedAtomsIndexes)

    print("usedAtoms[" + str(usedAtoms) + "]")
    print("usedAtomsCountMax[" + str(usedAtomsCountMax) + "]")
    print("total Treatments[" + str(len(parsedChemicalAnnotationSmiles_usedAtoms)) + "]")
    # calculate statistics about formulas (hystogram of used muleculas):
    # how many treatments use the atom - to be used for better random generation

    # read X plates avg well data (10)
    mean_well_profilesFileDF = readPlateData(mean_well_profilesFile)

    # split control wells and treated wells
    wellControl, wellTreatment = splitControlAndTreated(mean_well_profilesFileDF,
                                                        parsedChemicalAnnotationSmiles_usedAtoms_Hash)

    normalizedWellTreatment = normalizeTreatedWells(wellControl, wellTreatment)
    print(normalizedWellTreatment)

    # split train and test (10 cross validation)
    # Split a whole plate to be a validation
    # build model (wells to formula)
    # calculate RMSE per validation split
    print('end')


def readPlateData(mean_well_profilesFile):
    mean_well_profilesFileDF = pd.read_csv(mean_well_profilesFile);
    return mean_well_profilesFileDF


def splitControlAndTreated(mean_well_profilesFileDF, parsedChemicalAnnotationSmiles_usedAtoms_Hash):
    wellControl = mean_well_profilesFileDF.loc[mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    wellTreatment = mean_well_profilesFileDF.loc[~mean_well_profilesFileDF['Metadata_broad_sample'].isin(['DMSO'])]
    treatmentForLine = parsedChemicalAnnotationSmiles_usedAtoms_Hash[
        mean_well_profilesFileDF.iloc[0]['Metadata_pert_mfc_id']]
    return wellControl, wellTreatment


def normalizeTreatedWells(wellControl, wellTreatment):
    # Normalize treatment data using the control avarage
    # avg control data
    wellsDataStartColumnNumber = 17  # Cells_AreaShape_Area is the first numeric data column
    wellControlNumericData = wellControl.iloc[:, 17:]
    avgWellControlNumericData = wellControlNumericData.mean(axis=0)
    # reduce the avg from the actual data - to normalize
    normalizedWellTreatment = wellTreatment.apply(lambda x:
                                                  x[wellsDataStartColumnNumber:] - avgWellControlNumericData, axis=1)
    # set the treatment formula ID to the normalized treatments
    normalizedWellTreatment['Metadata_pert_mfc_id'] = wellTreatment.loc[:, 'Metadata_pert_mfc_id']
    return normalizedWellTreatment


run()
