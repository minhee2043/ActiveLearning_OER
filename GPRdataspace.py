import itertools as it
import numpy as np
from helperMethods import unique, count_metals, remove_zero_columns, multiplicity
from math import factorial
import os
from ase import Atoms
import sys
import csv
import ase
from ase.io import read
import pickle
from ase.calculators.vasp import Vasp2
from ase.io import read
from helperMethods import count_atoms, sortMetals
import itertools as it
#from slab import Slab
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Regression():
    def __init__(self):
        self.keepIds = []

    def weights(self, X, t, nMetals, zoneSizes):
        '''return optimized linear regression parameters (size: no. of features).
        Implemented from 'Learning from Data', Abu-Mostafa et al. p. 86, without bias parameter
        X        array (no. of samples x no. of features)   features
        t        array (no. of samples)                target values
        nMetals    int                                  number of metals in alloy
        zoneSizes  tuple of ints                       no. of atoms in each zone'''

        # remove columns that are pure zeros to avoid singular matrices later
        X, remIds, self.keepIds = remove_zero_columns(X)  # remIds has ids between 1 and 35
        n = len(remIds)

        # if any columns were removed
        if n == 1:
            print('feature number %d was zero for all samples and has been disregarded.'
                  % remIds[0])
            print('This features will thus not influence predictions.')
        elif n > 1:
            print('features numbered %s were zero for all samples and have been disregarded.'
                  % remIds.join(', '))
            print('These features will thus not influence predictions.')

        # calculate linear regression parameters
        XT = X.transpose()
        XTX = np.dot(XT, X)
        b = np.dot(XT, t)
        w = np.linalg.solve(XTX, b)

        ## center parameters around the average (except first zone)
        # number of adsorption ensembles
        nEns = factorial(nMetals + zoneSizes[0] - 1) / (factorial(zoneSizes[0]) * factorial(nMetals - 1))
        nEns=int(nEns)
        # number of zones except adsorption ensemble zone
        nGroups = (len(w) - nEns) / nMetals
        nGroups=int(nGroups)
        # initiate means of groups
        groupMeans = [0] * nGroups

        # loop through weights in groups of "nMetals
        for g in range(nGroups):
            start, stop = int(nEns + g * nMetals), int(nEns + (g + 1) * nMetals)
            ws = w[start:stop]

            # subtract mean from all parameters
            mean = sum(ws) / len(ws)
            ws = [w0 - mean for w0 in ws]
            groupMeans[g] = mean

            # update w
            w[start:stop] = ws

        # subtract group means from ensemble parameters
        for i in range(nEns):
            for j, mean in enumerate(groupMeans):
                w[i] += zoneSizes[j + 1] * mean

        return w

    def all_fingerprints(self, filename, nMetals, zoneSizes):
        '''write a csv file containing the number of each metal,
        the adsorption energy, and the multiplicity of all fingerprints.
        filename   String       name of csv file to save the output to
        w        list of floats linear regression parameters
        nMetals       int             number of metals in the alloy, e.g. 5
        zoneSizes  list of ints   number of atoms in the fingerprint zones, e.g. (1, 6, 3)'''

        # ensembles within each zone
        # e.g. [ [(0, ), (1, ), ...], 
        #      [(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1), ...],
        #      ... ]
        allZoneEns = [list(it.combinations_with_replacement(range(3), zoneSize)) for zoneSize in zoneSizes]
        print(allZoneEns)
        #sys.exit()
        # possible combinations for each zone
        # transform ensembles into counts of metals within each zone
        # e.g. [ [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ...],
        #      [[6, 0, 0, 0, 0], [5, 1, 0, 0, 0], ...],
        #      ...]
        zoneCounts = [[count_metals(ens, nMetals) for ens in zoneEnss] for zoneEnss in allZoneEns] 
        # for all possible comb of zones, count num of metals for each element
        # number of ensembles of each zone
        # e.g. [5, 210, 35]
        nZoneEns = [len(zoneEnss) for zoneEnss in allZoneEns]

        # number of lines of csv file
        # e.g. 36,750
        nLines = np.prod(nZoneEns)
        # total combinations

        # if lots of fingerprints are saved then save each adsorption ensemble as individual files
        saveEns = False
        if nLines > 1e7:
            saveEns = True

            # redefine number of lines per file
            nLines = nLines // nZoneEns[0]

        # initiate arrays
        counts = np.zeros((nLines, nMetals))
        energies = np.zeros(nLines)
        mults = np.zeros(nLines)
        feature=np.zeros((nLines,24))
        # initiate countera
        i = 0
    
        # loop through all adsorption site ensembles
        for adsEnsId, adsEns in enumerate(allZoneEns[0]):

            if saveEns:
                # reset arrays
                counts = np.zeros((nLines, nMetals), dtype='int64')
                energies = np.zeros(nLines, dtype='float')
                mults = np.zeros(nLines, dtype='int64')

                # reset counter
                i = 0

            # adsorption ensemble fingerprint
            # e.g. [1, 0, 0, 0, 0]
            adsFp = [0] * nZoneEns[0]
            adsFp[adsEnsId] = 1

            # count metals in adsorption ensembe
            # e.g. [1, 0, 0, 0, 0]
            adsCount = count_metals(adsEns, nMetals)

            # multiplicity of adsorption ensemble
            # e.g. 1
            adsMult = multiplicity(zoneSizes[0], adsCount)

            # loop through all remaining zones, generated by counting the metals
            # e.g. [[6, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
            for theseZoneCounts in it.product(*zoneCounts[1:]):
                # link zone counts together
                # e.g. [6, 0, 0, 0, 0, 3, 0, 0, 0, 0]
                zoneFp = list(it.chain.from_iterable(theseZoneCounts))

                # append to adsorption fingerprint to make the final fingerprint
                
                fp = np.array(adsFp + zoneFp) 
                feature[i]=fp
                # predict energy
                #energies[i] = self.predicted_energies(fp, w)
                #print(energies[i])
                # count of metals of fingerprint
                # e.g. [10, 0, 0, 0, 0]
                counts[i] = (np.array(adsCount) +
                             sum(np.array(thisZoneCounts) for thisZoneCounts in theseZoneCounts))

                # multiplicity of fingerprint
                # e.g. 1
                mults[i] = (adsMult *
                            np.prod([multiplicity(zoneSize, zoneCount)
                                     for zoneSize, zoneCount in zip(zoneSizes[1:], theseZoneCounts)]))

                # increment counter
                i += 1

            if saveEns:
                output = np.c_[feature, mults]

                # add an underscored number of the current ensemble in the filename
                parts = filename.rpartition('.')
                fname = parts[0] + '_%d' % adsEnsId + parts[1] + parts[2]

                # save file for the current ensemble
                np.savetxt(fname, output, fmt=['%d'] * 24 + ['%d'], delimiter=',')
                print('ensemble %d saved' % adsEnsId)

        if not saveEns:
            # save csv
            output = np.c_[feature, mults]
            np.savetxt(filename, output, fmt=['%d'] * 24 +['%d'], delimiter=',')


    def predicted_energies(self, X, w):
        '''return list of predicted energies (size: no. of samples)
        X (no. of samples x no. of features)   feature array
        w (no. of features)                   linear regression parameters'''

        # if the number of features is greater than the number of parameters
        if len(X.T) > len(w):

            # if only one sample
            if X.ndim == 1:

                # keep only columns that have defined parameters
                X = X[self.keepIds]
            else:

                # keep only columns that have defined parameters
                X = X[:, self.keepIds]

        # return predicted energies
        return np.dot(X, w)

