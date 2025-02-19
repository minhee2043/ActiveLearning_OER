from ase import Atoms
from ase.visualize import view
from ase.build import bulk, molecule, fcc111
from ase.calculators.vasp import Vasp
from matplotlib import pyplot as plt
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import numpy as np
import copy
from itertools import combinations, product
from ase.io import Trajectory, read, write
import ase.db
from ase.db import connect
from ase.build import add_adsorbate
import random
import sys
from collections import defaultdict
from helperMethods import multiplicity  # Custom helper function
import csv
import fileinput

# Initialize arrays for storing results
i = 0
mults = np.zeros(6561)  # 3^8 possible combinations (3 metals, 8 positions)
feature = np.zeros((6561, 21))  # Features matrix

"""
Generate all possible surface configurations. In this case, 2*2*4 structure
Two layers are repeated twice for simplicity as only the top two layers are important for featurization
Each position can be Ni, Co, or Fe
"""
possible_surface = list(product(['Ni', 'Co', 'Fe'], repeat=8))

# Iterate through all possible surface configurations
for comb in possible_surface:
    # Initialize dictionaries to count metal atoms in different positions
    # These represent different coordination environments or positions in the surface
    first = {'Ni': 0, 'Co': 0, 'Fe': 0}
    second = {'Ni': 0, 'Co': 0, 'Fe': 0}
    third = {'Ni': 0, 'Co': 0, 'Fe': 0}
    fourth = {'Ni': 0, 'Co': 0, 'Fe': 0}
    fifth = {'Ni': 0, 'Co': 0, 'Fe': 0}

    # Count metals in different positions based on the combination
    # The indices correspond to specific positions in the surface structure
    fifth[comb[0]]+=1
    fifth[comb[1]]+=1
    fifth[comb[2]]+=1
    third[comb[2]]+=1
    third[comb[3]]+=2
    first[comb[4]]+=1
    second[comb[4]]+=2
    first[comb[5]]+=1
    second[comb[5]]+=2
    first[comb[6]]+=1
    fourth[comb[6]]+=1
    second[comb[7]]+=2
    fourth[comb[7]]+=2
    
    # Convert dictionaries to lists of values
    firstvalues = list(first.values())
    secondvalues = list(second.values())
    thirdvalues = list(third.values())
    fourthvalues = list(fourth.values())
    fifthvalues = list(fifth.values())
    
    # Filter out zero values (positions where metal is not present)
    firstval = [x for x in firstvalues if x != 0]
    secondval = [x for x in secondvalues if x != 0]
    thirdval = [x for x in thirdvalues if x != 0]
    fourthval = [x for x in fourthvalues if x != 0]
    fifthval = [x for x in fifthvalues if x != 0]
    
    # Calculate multiplicities for each position
    # The first number in each multiplicity call represents the weight of that position
    firstmult = multiplicity(2, firstval)
    secondmult = multiplicity(4, secondval)
    thirdmult = multiplicity(2, thirdval)
    fourthmult = multiplicity(2, fourthval)
    fifthmult = multiplicity(1, fifthval)
    
    # Calculate total multiplicity by multiplying individual multiplicities
    totalmult = firstmult * secondmult * thirdmult * fourthmult * fifthmult 
    
    # Store features and multiplicity
    feature[i] = np.array(firstvalues + secondvalues + thirdvalues + 
                         fourthvalues + fifthvalues )
    mults[i] = totalmult
    i += 1

# Save surface configurations to CSV
file = open('index_metal.csv', 'w', newline='')
with file:
    write = csv.writer(file)
    write.writerows(possible_surface)

# Combine features and multiplicities and save to CSV
output = np.c_[feature, mults]
np.savetxt('possibleFp.csv', output, fmt=['%d']*16, delimiter=',')
