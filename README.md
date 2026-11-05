# Active Learning for OER Alloy Catalyst Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **This repository contains the code for the paper:**
> H. Jeon, M. Park, W. B. Lee, S. Park, Y. Kim, S. Mhin,
> "[Accelerated Discovery of Earth-Abundant Oxygen Evolution Catalysts Through Surface-Informed Active Learning](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.78281)",
> *Advanced Science*, e78281 (2026). DOI: [10.1002/advs.78281](https://doi.org/10.1002/advs.78281)
>
> If you use this code, please cite the paper (see [Citation](#citation)).

An active learning framework for efficiently navigating the chemical space of ternary alloy catalysts for the Oxygen Evolution Reaction (OER). This repository implements Gaussian Process Regression (GPR) with intelligent query strategies to minimize the number of expensive DFT calculations needed to identify promising catalyst compositions.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [File Descriptions](#file-descriptions)
- [Data Format](#data-format)
- [Customization Guide](#customization-guide)
- [Citation](#citation)
- [License](#license)

---

## Overview

### The Problem
Discovering optimal alloy catalysts for OER requires screening vast compositional spaces. Traditional approaches require DFT calculations for every candidate composition, which is computationally prohibitive.
### Our Solution
This active learning framework:
1. **Generates a complete feature space** representing all possible surface configurations
2. **Intelligently selects** which configurations to calculate via DFT using acquisition functions
3. **Predicts energies** for uncalculated configurations using Gaussian Process Regression
4. **Iteratively improves** predictions by selecting the most informative next calculations
5. **Estimates catalyst activity** using Boltzmann-weighted kinetic models

### System Details
- **Target system**: Ni-Fe-Co ternary alloy catalysts
- **Adsorbates**: O and OH intermediates
- **Surface sites**: fcc and hcp hollow sites
- **Feature space**: 280,000 unique surface motifs
- **Feature vector**: 15-element fingerprint (3 metals × 5 coordination zones)

---

## Key Features

✅ **Comprehensive feature generation** - Automated fingerprint creation for all possible surface configurations  
✅ **Intelligent sampling** - Active learning acquisition functions to minimize DFT calculations  
✅ **Gaussian Process Regression** - Uncertainty-aware energy predictions  
✅ **Periodic boundary handling** - DFT-compatible dataspace generation  
✅ **Activity estimation** - Boltzmann distribution-based catalytic activity calculation  
✅ **Extensible framework** - Easily adaptable to other alloy systems and reactions  

---

## Installation

### Prerequisites
- Python 3.8 or higher
- ASE (Atomic Simulation Environment) for structure handling

### Required Dependencies
```bash
pip install numpy>=1.21.2 scipy>=1.7.1 pandas>=1.3.3 matplotlib>=3.4.3 scikit-learn>=0.24.2
```

### Clone the Repository
```bash
git clone https://github.com/minhee2043/ActiveLearning_OER.git
cd ActiveLearning_OER
```

---

## Quick Start

### Complete Workflow (5 Steps)

```bash
# Step 1: Generate complete feature space (280,000 configurations)
python GPRdataspace.py

# Step 2: Generate DFT-compatible subset for calculations
python possibleFp.py

# Step 3: Run active learning cycles (GPR + acquisition function)
python mygaussian.py

# Step 4: Preprocess data for activity calculation
python sum_element.py

# Step 5: Calculate and visualize catalyst activity
python activity_plot.py
```

### Converting DFT Structures to Features

For analyzing individual DFT-relaxed structures:

```python
from motif_to_feature import Slab
import ase.io
import numpy as np

# Read DFT-relaxed structure
trajectory = ase.io.read('relaxed_structure.traj')

# Extract surface motif
motif = Slab(trajectory)

# Generate 15-element feature vector
# metals: reference order for counting
# zones: coordination shells around adsorption site
feature = np.array(motif.features(
    metals=['Ni', 'Fe', 'Co'],
    zones=['ens', 'sn', 'ssn', 'sf', 'ssf']
))

print(f"Feature vector: {feature}")
# Output: [Ni1,Fe1,Co1, Ni2,Fe2,Co2, Ni3,Fe3,Co3, Ni4,Fe4,Co4, Ni5,Fe5,Co5]
```

---

## Detailed Workflow

### Phase 1: Feature Space Generation

#### 1.1 Complete Dataspace (`GPRdataspace.py`)
Generates all 280,000 possible surface configurations for the Ni-Fe-Co ternary system.

**Input**: None (uses default parameters: 3 metals, 5 zones)  
**Output**: `GPRdataspace.csv` - 280,000 rows × 16 columns (15 features + 1 multiplicity)  
**Key parameters** (modifiable in code):
```python
nMetals = 3              # Number of metal types
zoneSizes = (3, 6, 3, 3, 3)  # Atoms in each coordination shell
```

#### 1.2 DFT-Compatible Subset (`possibleFp.py`)
Filters the complete dataspace to only configurations compatible with periodic DFT calculations.

**Input**: `GPRdataspace.csv`  
**Output**: `possibleFp.csv` - Subset of valid DFT configurations  

---

### Phase 2: Active Learning Cycles

#### 2.1 Gaussian Process Regression (`mygaussian.py`)
Core active learning engine that iteratively selects calculations and builds predictive models.

**Inputs**:
- DFT training data (user-provided)
- `possibleFp.csv` (candidate pool)

**Outputs**:
- Trained GPR models for each iteration
- Predicted energies for all configurations
- Selected next calculation points from the candidate pool

---

### Phase 3: Activity Calculation

#### 3.1 Data Preprocessing (`sum_element.py`)
Aggregates features by total composition for activity estimation.

**Input**: Predicted energies for all configurations  
**Output**: Composition-aggregated data  

#### 3.2 Activity Calculation (`activity_plot.py`)
Estimates catalytic activity using microkinetic modeling.

**Input**: Composition-aggregated energies  
**Output**: 
- Activity maps

**Key equations**:
```
Rate ∝ Σ_i w_i × exp(-ΔG_i / kT)
where w_i = multiplicity, ΔG_i = reaction free energy
```

---

## File Descriptions

### Core Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `GPRdataspace.py` | Generate complete feature space | Parameters | `GPRdataspace.csv` |
| `possibleFp.py` | Filter DFT-compatible configurations | `GPRdataspace.csv` | `possibleFp.csv` |
| `motif_to_feature.py` | Convert structures to features | ASE Atoms | 15-element vector |
| `mygaussian.py` | Active learning engine | DFT data, `possibleFp.csv` | Predictions, next points |
| `sum_element.py` | Preprocess for activity | Predicted energies | Aggregated data |
| `activity_plot.py` | Calculate activity | Aggregated energies | Plots, rankings |

### Helper Modules

| File | Purpose |
|------|---------|
| `helperMethods.py` | Mathematical utilities (count_atoms, multiplicity, etc.) |

---

## Data Format

### Input Data Formats

#### DFT Trajectory Files (`.traj`)
- **Format**: ASE trajectory format
- **Required tags**:
  - `tag = 0`: Adsorbate atoms (O, OH)
  - `tag = 1`: Surface layer atoms
  - `tag = 2`: Subsurface layer atoms
  - `tag = 3`: Third layer atoms
  - `tag = 4`: Bottom layer atoms
- **Atoms**: Must contain Ni, Fe, Co metals

#### Training Data CSV
Expected format for initial DFT data:
```csv
Ni1,Fe1,Co1,Ni2,Fe2,Co2,Ni3,Fe3,Co3,Ni4,Fe4,Co4,Ni5,Fe5,Co5,Energy
3,0,0,6,0,0,3,0,0,6,0,0,3,0,0,-4.52
2,1,0,5,1,0,2,1,0,5,1,0,2,1,0,-4.38
...
```

### Output Data Formats

#### `GPRdataspace.csv`
```csv
# 280,000 rows × 16 columns
# Columns 1-15: Feature vector (metal counts in 5 zones)
# Column 16: Multiplicity (degeneracy factor)
3,0,0,6,0,0,3,0,0,6,0,0,3,0,0,1
2,1,0,6,0,0,3,0,0,6,0,0,3,0,0,3
...
```

#### `possibleFp.csv`
Subset of `GPRdataspace.csv` with only DFT-compatible configurations.

#### Prediction Output
```csv
Ni1,Fe1,Co1,Ni2,Fe2,Co2,Ni3,Fe3,Co3,Ni4,Fe4,Co4,Ni5,Fe5,Co5,Predicted_Energy,Uncertainty
3,0,0,6,0,0,3,0,0,6,0,0,3,0,0,-4.50,0.15
...
```

---

## Customization Guide

### For Different Alloy Systems

#### Change Metal Types
**File**: `GPRdataspace.py`, `motif_to_feature.py`

```python
# Original: Ni-Fe-Co system
metals = ['Ni', 'Fe', 'Co']

# Example: Pt-Pd-Rh system
metals = ['Pt', 'Pd', 'Rh']
```

#### Change Number of Metals
**File**: `GPRdataspace.py`

```python
# Original: ternary system
nMetals = 3
range(3)  # in combinations_with_replacement

# Example: quaternary system (4 metals)
nMetals = 4
range(4)  # update this line
```

#### Modify Coordination Zones
**File**: `GPRdataspace.py`

```python
# Original: 5 zones with (3,6,3,3,3) atoms
zoneSizes = (3, 6, 3, 3, 3)

# Example: 3 zones for simpler fingerprint
zoneSizes = (1, 6, 3)  # 1 on-top + 6 neighbors + 3 subsurface

```

**File**: `motif_to_feature.py`

```python
# Update zone list accordingly
zones = ['ens', 's', 'ss']  # for 3-zone system
```

### For Different Adsorption Sites

#### On-Top Adsorption
**File**: `motif_to_feature.py`

```python
# Automatic detection
is_ontop = motif.onTop()
feature = motif.features(['Ni','Fe','Co'], onTop=is_ontop, zones=['ens','s','ss'])

# Manual specification
feature = motif.features(['Ni','Fe','Co'], onTop=True, zones=['ens','s','ss'])
```

#### Different Zone Definitions
Create custom zone methods in `Slab` class:

```python
def custom_zone(self, onTop=False):
    '''Your custom coordination zone definition'''
    if onTop:
        return self.closest(layer=1, start=X, stop=Y)[0]
    else:
        return self.closest(layer=1, start=A, stop=B)[0]
```

### For Different Acquisition Functions

**File**: `mygaussian.py`

```python
# Upper Confidence Bound (default)
def acquisition_ucb(mu, sigma, kappa=2.0):
    return mu + kappa * sigma

# Expected Improvement
def acquisition_ei(mu, sigma, y_best):
    Z = (mu - y_best) / sigma
    return (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

# Probability of Improvement
def acquisition_pi(mu, sigma, y_best):
    Z = (mu - y_best) / sigma
    return norm.cdf(Z)
```

### For Different Kernels

**File**: `mygaussian.py`

```python
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

# Original: RBF kernel
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# Alternative: Matern kernel (more flexible)
kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

# Alternative: Rational Quadratic (scale-invariant)
kernel = RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=0.1)
```

---

## Citation
If you find this code useful in your research, please cite:
```bibtex
@article{jeon2026accelerated,
  title   = {Accelerated Discovery of Earth-Abundant Oxygen Evolution Catalysts Through Surface-Informed Active Learning},
  author  = {Jeon, Hayun and Park, Minhee and Lee, Won Bo and Park, Seyoung and Kim, YongJoo and Mhin, Sungwook},
  journal = {Advanced Science},
  pages   = {e78281},
  year    = {2026},
  doi     = {10.1002/advs.78281}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [https://github.com/minhee2043/ActiveLearning_OER/issues](https://github.com/minhee2043/ActiveLearning_OER/issues)
- **Email**: minhee2043@snu.ac.kr


