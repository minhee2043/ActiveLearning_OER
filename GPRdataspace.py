import numpy as np
from helperMethods import unique, count_metals, remove_zero_columns, multiplicity
from math import factorial
import os
import csv
import pickle
from helperMethods import count_atoms, sortMetals
import itertools as it
import pandas as pd

class Regression():
    def __init__(self):
        self.keepIds = []

    def all_fingerprints(self, filename, nMetals, zoneSizes):
        '''Generate all possible fingerprints for surface alloy configurations and save to CSV.
        
        Example: For NiFeCo alloy with zones (3, 6, 3, 3, 3):
        - Creates 280,000 unique fingerprints (10 × 28 × 10 × 10 × 10)
        - Each fingerprint has 16 columns: 15 for metal counts + 1 for multiplicity
        - Metal counts organized as [Ni1,Fe1,Co1, Ni2,Fe2,Co2, ..., Ni5,Fe5,Co5]
        
        Parameters:
        filename   String       CSV file to save fingerprints, e.g. 'fingerprints.csv'
        nMetals    int          Number of metal types (must be 3 for Ni, Fe, Co)
        zoneSizes  tuple of ints  Number of atoms in each zone, e.g. (3, 6, 3, 3, 3)
        '''

        # Generate all possible atomic arrangements (ensembles) for each zone
        # range(3) represents 3 metal types: 0=Ni, 1=Fe, 2=Co
        # For zoneSizes = (3, 6, 3, 3, 3):
        # Zone 1 (3 atoms): [(0,0,0), (0,0,1), (0,0,2), (0,1,1), ...] → 10 ensembles
        # Zone 2 (6 atoms): [(0,0,0,0,0,0), (0,0,0,0,0,1), ...] → 28 ensembles
        # Zones 3,4,5 (3 atoms each): 10 ensembles each
        allZoneEns = [list(it.combinations_with_replacement(range(3), zoneSize)) for zoneSize in zoneSizes]
        
        # Convert each ensemble to metal counts [Ni_count, Fe_count, Co_count]
        # For example:
        # Zone 1: [[3,0,0], [2,1,0], [2,0,1], [1,2,0], [1,1,1], [1,0,2], [0,3,0], [0,2,1], [0,1,2], [0,0,3]]
        # Zone 2: [[6,0,0], [5,1,0], [5,0,1], ..., [0,0,6]] (28 different counts)
        # Zones 3,4,5: Same as Zone 1 (10 different counts each)
        zoneCounts = [[count_metals(ens, nMetals) for ens in zoneEnss] for zoneEnss in allZoneEns] 
        
        # Number of unique ensembles in each zone
        # For (3, 6, 3, 3, 3): nZoneEns = [10, 28, 10, 10, 10]
        nZoneEns = [len(zoneEnss) for zoneEnss in allZoneEns]

        # Total number of fingerprints = product of all zone ensemble counts
        # For (3, 6, 3, 3, 3): nLines = 10 × 28 × 10 × 10 × 10 = 280,000
        nLines = np.prod(nZoneEns)

        # If more than 10 million fingerprints, save as separate files (one per zone 1 ensemble)
        saveEns = False
        if nLines > 1e7:
            saveEns = True
            # Each file will contain fingerprints for one zone 1 ensemble
            nLines = nLines // nZoneEns[0]

        # Initialize arrays to store fingerprints
        counts = np.zeros((nLines, nMetals), dtype='int64')  # Total metal counts (not used in output)
        mults = np.zeros(nLines, dtype='int64')  # Multiplicity values
        feature = np.zeros((nLines, 3*len(zoneSizes)), dtype='int64')  # Fingerprint: 3 metals × 5 zones = 15 elements
        
        # Counter for current row
        i = 0
    
        # Loop through all zone 1 ensembles (10 ensembles for 3 atoms)
        for adsEnsId, adsEns in enumerate(allZoneEns[0]):

            if saveEns:
                # Reset arrays for new file (one file per zone 1 ensemble)
                counts = np.zeros((nLines, nMetals), dtype='int64')
                mults = np.zeros(nLines, dtype='int64')
                feature = np.zeros((nLines, 3*len(zoneSizes)), dtype='int64')

                # Reset counter
                i = 0

            # Count metals in zone 1 ensemble
            # For example, if adsEns = (0,0,1): adsCount = [2, 0, 1] meaning 2 Ni, 0 Fe, 1 Co
            adsCount = count_metals(adsEns, nMetals)

            # Calculate multiplicity for zone 1 ensemble
            # Multiplicity = number of ways to arrange atoms with given composition
            # For [2,0,1]: mult = 3!/(2!×0!×1!) = 3
            adsMult = multiplicity(zoneSizes[0], adsCount)

            # Loop through all combinations of zones 2-5
            # Creates cartesian product: zone2 × zone3 × zone4 × zone5
            # For (3,6,3,3,3): 28 × 10 × 10 × 10 = 28,000 combinations per zone 1 ensemble
            for theseZoneCounts in it.product(*zoneCounts[1:]):
                # Flatten the metal counts from zones 2-5 into a single list
                # For example: [[5,1,0], [2,1,0], [3,0,0], [1,1,1]] → [5,1,0,2,1,0,3,0,0,1,1,1]
                zoneFp = list(it.chain.from_iterable(theseZoneCounts))

                # Combine zone 1 counts with zones 2-5 counts to create complete fingerprint
                # Final fingerprint format: [Ni1,Fe1,Co1, Ni2,Fe2,Co2, Ni3,Fe3,Co3, Ni4,Fe4,Co4, Ni5,Fe5,Co5]
                # Example: [2,0,1, 5,1,0, 2,1,0, 3,0,0, 1,1,1] = 15 elements
                fp = np.array(list(adsCount) + zoneFp)
                feature[i] = fp
                
                # Calculate total metal counts across all zones (for verification, not used in output)
                # Example: [2,0,1] + [5,1,0] + [2,1,0] + [3,0,0] + [1,1,1] = [13, 3, 2]
                counts[i] = (np.array(adsCount) +
                             sum(np.array(thisZoneCounts) for thisZoneCounts in theseZoneCounts))

                # Calculate total multiplicity = zone1_mult × zone2_mult × zone3_mult × zone4_mult × zone5_mult
                # Example: 3 × 6 × 3 × 1 × 3 = 162
                mults[i] = (adsMult *
                            np.prod([multiplicity(zoneSize, zoneCount)
                                     for zoneSize, zoneCount in zip(zoneSizes[1:], theseZoneCounts)]))

                # Move to next row
                i += 1

            if saveEns:
                # Combine fingerprints (15 columns) with multiplicity (1 column) = 16 columns total
                output = np.c_[feature, mults]

                # Create filename with zone 1 ensemble ID appended
                # Example: 'fingerprints.csv' → 'fingerprints_0.csv', 'fingerprints_1.csv', ...
                parts = filename.rpartition('.')
                fname = parts[0] + '_%d' % adsEnsId + parts[1] + parts[2]

                # Save file with 15 integer columns (fingerprint) + 1 integer column (multiplicity)
                np.savetxt(fname, output, fmt=['%d'] * (3*len(zoneSizes)) + ['%d'], delimiter=',')
                print('ensemble %d saved' % adsEnsId)

        if not saveEns:
            # Combine all fingerprints into single output array
            # Shape: (280000, 16) for zoneSizes=(3,6,3,3,3)
            output = np.c_[feature, mults]
            
            # Save complete fingerprint database as CSV
            # Format: 15 columns of integers (metal counts) + 1 column of integers (multiplicity)
            np.savetxt(filename, output, fmt=['%d'] * (3*len(zoneSizes)) + ['%d'], delimiter=',')



