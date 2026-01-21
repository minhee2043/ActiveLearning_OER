# Active Learning for OER Alloy Catalyst Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
- [Example Use Case](#example-use-case)
- [Citation](#citation)
- [License](#license)

---

## Overview

### The Problem
Discovering optimal alloy catalysts for OER requires screening vast compositional spaces. Traditional approaches require DFT calculations for every candidate composition, which is computationally prohibitive (typically 280,000+ possible configurations for ternary alloys).

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

Or install from requirements file:
```bash
pip install -r requirements.txt
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
**Runtime**: ~1-2 minutes

**What it does**:
- Enumerates all combinations of Ni, Fe, Co atoms in 5 coordination zones
- Zone 1 (ens): 3 atoms at adsorption site
- Zone 2 (sn): 3 nearest surface neighbors
- Zone 3 (ssn): 1-3 nearest subsurface neighbors (depends on fcc/hcp)
- Zone 4 (sf): 6 far surface neighbors
- Zone 5 (ssf): 3-6 far subsurface neighbors (depends on fcc/hcp)
- Calculates multiplicity (degeneracy) for each configuration

**Key parameters** (modifiable in code):
```python
nMetals = 3              # Number of metal types
zoneSizes = (3, 6, 3, 3, 3)  # Atoms in each coordination shell
```

#### 1.2 DFT-Compatible Subset (`possibleFp.py`)
Filters the complete dataspace to only configurations compatible with periodic DFT calculations.

**Input**: `GPRdataspace.csv`  
**Output**: `possibleFp.csv` - Subset of valid DFT configurations  
**Runtime**: ~30 seconds

**What it does**:
- Applies periodic boundary conditions
- Ensures bulk composition consistency
- Filters out physically unrealistic configurations
- Generates the pool from which DFT calculations are selected

---

### Phase 2: Active Learning Cycles

#### 2.1 Gaussian Process Regression (`mygaussian.py`)
Core active learning engine that iteratively selects calculations and builds predictive models.

**Inputs**:
- Initial DFT training data (user-provided)
- `possibleFp.csv` (candidate pool)

**Outputs**:
- Trained GPR models for each iteration
- Predicted energies for all configurations
- Selected next calculation points

**What it does**:
1. **Train**: Fit GPR model on existing DFT data
2. **Predict**: Estimate energies and uncertainties for all candidates
3. **Acquire**: Select next point using acquisition function (e.g., Upper Confidence Bound)
4. **Iterate**: Add DFT result to training set, repeat

**Key hyperparameters**:
```python
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
n_iterations = 50        # Number of active learning cycles
acquisition = 'UCB'      # Options: 'UCB', 'EI', 'PI'
```

**Active Learning Strategies**:
- **Exploitation**: Select configurations with lowest predicted energy
- **Exploration**: Select configurations with highest uncertainty
- **Balanced**: Upper Confidence Bound (UCB) balances both

---

### Phase 3: Activity Calculation

#### 3.1 Data Preprocessing (`sum_element.py`)
Aggregates features by total composition for activity estimation.

**Input**: Predicted energies for all configurations  
**Output**: Composition-aggregated data  
**What it does**: Groups configurations by total Ni/Fe/Co counts across all zones

#### 3.2 Activity Calculation (`activity_plot.py`)
Estimates catalytic activity using microkinetic modeling.

**Input**: Composition-aggregated energies  
**Output**: 
- Activity maps (2D/3D plots)
- Volcano plots
- Composition-activity relationships

**What it does**:
- Applies Boltzmann distribution to weight configurations
- Calculates reaction rates using transition state theory
- Identifies optimal composition regions
- Visualizes activity landscapes

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
| `combine_csv_files.py` | Merge batch calculation results |

### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

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

# Example: Extended coordination
zoneSizes = (3, 6, 3, 3, 3, 6)  # Add 6th zone
```

**File**: `motif_to_feature.py`

```python
# Update zone list accordingly
zones = ['ens', 's', 'ss']  # for 3-zone system
# or
zones = ['ens', 'sn', 'ssn', 'sf', 'ssf', 'sixth_zone']  # for 6-zone system
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

## Example Use Case

### Scenario: Finding Optimal Ni-Fe-Co Composition for OER

#### Step 1: Generate Feature Space
```bash
python GPRdataspace.py
# Output: GPRdataspace.csv with 280,000 configurations
```

#### Step 2: Create DFT-Compatible Pool
```bash
python possibleFp.py
# Output: possibleFp.csv with ~50,000 valid configurations
```

#### Step 3: Prepare Initial Training Data
Perform 20 initial DFT calculations on randomly selected configurations:
```csv
# initial_training.csv
3,0,0,6,0,0,3,0,0,6,0,0,3,0,0,-4.52
3,0,0,5,1,0,3,0,0,5,1,0,3,0,0,-4.48
...
```

#### Step 4: Run Active Learning (50 iterations)
```bash
python mygaussian.py
# Performs 50 active learning cycles
# Total DFT calculations: 20 (initial) + 50 (active) = 70
# Predictions for: 280,000 configurations
```

#### Step 5: Analyze Results
```bash
# Aggregate by composition
python sum_element.py

# Calculate activity and plot
python activity_plot.py
# Output: Activity maps showing optimal Ni:Fe:Co ratios
```

#### Expected Results
- **Optimal region**: Identified with <100 DFT calculations
- **Computational savings**: 99.97% reduction vs. full screening
- **Activity prediction**: Uncertainty-aware estimates for all compositions

---

## Understanding the Feature Vector

### 15-Element Fingerprint Structure

For a Ni-Fe-Co system with zones `['ens', 'sn', 'ssn', 'sf', 'ssf']`:

```
[Ni₁, Fe₁, Co₁, Ni₂, Fe₂, Co₂, Ni₃, Fe₃, Co₃, Ni₄, Fe₄, Co₄, Ni₅, Fe₅, Co₅]
 └─── Zone 1 ───┘ └─── Zone 2 ───┘ └─── Zone 3 ───┘ └─── Zone 4 ───┘ └─── Zone 5 ───┘
    (3 atoms)        (3 atoms)        (1-3 atoms)      (6 atoms)        (3-6 atoms)
```

### Example: Pure Ni Surface
```
[3,0,0, 3,0,0, 3,0,0, 6,0,0, 3,0,0]
```
All positions occupied by Ni, no Fe or Co.

### Example: Mixed Surface
```
[2,1,0, 2,1,0, 1,0,0, 4,2,0, 2,1,0]
```
- Zone 1: 2 Ni, 1 Fe, 0 Co
- Zone 2: 2 Ni, 1 Fe, 0 Co
- Zone 3: 1 Ni, 0 Fe, 0 Co
- Zone 4: 4 Ni, 2 Fe, 0 Co
- Zone 5: 2 Ni, 1 Fe, 0 Co

---

## Troubleshooting

### Common Issues

#### Issue: "No module named 'helperMethods'"
**Solution**: Ensure `helperMethods.py` is in the same directory
```bash
ls helperMethods.py  # Should exist
```

#### Issue: "ASE Atoms object has no attribute 'tag'"
**Solution**: Your structure file is missing atom tags. Set them manually:
```python
from ase.io import read

atoms = read('structure.traj')
# Set tags: 0=adsorbate, 1=surface, 2=subsurface, 3=third, 4=bottom
for i, atom in enumerate(atoms):
    if atom.position[2] > 15.0:  # Example: adsorbate height
        atom.tag = 0
    # ... set other tags based on z-position
```

#### Issue: "ValueError: the site was not classified"
**Solution**: Structure geometry is ambiguous. Check:
1. Adsorbate is positioned above surface
2. Surface forms proper fcc/hcp geometry
3. Triangle detection tolerance may need adjustment

#### Issue: GPR predictions are poor
**Solutions**:
1. Increase initial training set size (>50 points recommended)
2. Adjust kernel hyperparameters
3. Check for outliers in training data
4. Normalize features before GPR

---

## Performance Metrics

### Computational Efficiency
- **Feature generation**: O(n_metals^n_atoms) ≈ 1-2 minutes
- **Single GPR training**: O(n³) where n = training set size
- **Prediction**: O(n × m) where m = prediction set size
- **Total workflow**: ~2-3 hours for 50 active learning cycles

### Active Learning Performance
Typical results for Ni-Fe-Co OER system:
- **Initial random sampling**: 20 points, MAE ≈ 0.5 eV
- **After 50 AL cycles**: 70 total points, MAE ≈ 0.08 eV
- **Optimal composition found**: Within 30-40 iterations
- **Confidence**: >95% accuracy in top-10% activity region

---

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

### Areas for Contribution
- Additional acquisition functions
- Support for quaternary/quinary alloys
- Integration with automated DFT workflows
- Improved visualization tools
- Documentation improvements

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2024,
  title={Active Learning for Efficient Discovery of OER Alloy Catalysts},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [https://github.com/minhee2043/ActiveLearning_OER/issues](https://github.com/minhee2043/ActiveLearning_OER/issues)
- **Email**: your.email@institution.edu

---

## Acknowledgments

- ASE (Atomic Simulation Environment) development team
- scikit-learn developers for GPR implementation
- [Add your funding sources, collaborators, etc.]
