# ActiveLearning_OER
Active learning based model to sufficiently navigate OER alloy catalyst

This repository contains the implementation of an active learning model to navigate OER catalyst. The codes are organized to generate features, preprocess data, and perform Gaussian process regression to predict energies and select the next points for calculation.

## Usage
Clone the repository: git clone https://github.com/your-repo.git
cd your-repo

Install the required dependencies: 
The following dependencies are required to run the scripts in this repository. 
```
numpy>=1.21.2
scipy>=1.7.1
pandas>=1.3.3
matplotlib>=3.4.3
scikit-learn>=0.24.2
```

Run the scripts in the following order

Generate features for the surface motif:
```
python GPRdataspace.py
```
Generate DFT-calculated dataspace:
```
python possibleFp.py
```
Convert surface motif to feature vector:
```
from motif_to_feature import Slab
```
Preprocess data for activity calculation:
```
python sum_element.py
```
Calculate activity based on Boltzmann distribution:
```
python activity_plot.py
```
Assess the model with the final dataset:
```
python parity_plot.py
```

## Data Format
-Input Data:
  -Surface motifs and their corresponding features.
  -DFT-calculated dataspace.
  -Preprocessed data for activity calculation.

-Output Data:
  -GPR predicted energies by the feature vectors for surface motifs.
  -Activity calculations based on Boltzmann distribution.
  -Model assessment plots.

## Explanation of the Codes
**GPRdataspace.py**
Generates all the features for the surface motif. This represents the total dataspace that requires prediction of energies.

**possibleFp.py**
Due to the periodic boundary condition of the plane-wave basis set, a DFT-calculated dataspace is generated. The candidates for DFT-calculation are selected within this dataspace.

**helperMethods.py**
Contains simple mathematical functions that help the featurization process.

**motif_to_feature.py**
Converts the surface motif of the adsorption site to a feature vector.

**mygaussian.py**
Our active learning model. Contains Gaussian process regression model and acquisition function for selecting the next calculated points.

**sum_element.py**
Preprocessing of data for activity calculation. For each feature, the number of atoms of three elements are summed.

**activity_plot.py**
Based on the Boltzmann distribution of reaction kinetics, the activity of ternary catalyst is calculated.

**parity_plot.py**
Assessment of the model with the final dataset.

## Modifications for Other Users
**Feature Generation**: Modify GPRdataspace.py to include additional features or change the existing feature generation logic.

**DFT-calculated Dataspace**: Update possibleFp.py to change the alloy conditions

**Surface Motif to Feature Vector**: Adjust motif_to_feature.py to change how surface motifs are converted to feature vectors.

**Active Learning Model**: Customize mygaussian.py to change the Gaussian process regression model or the acquisition function. Filename should be edtited.

**Model Assessment**: Adjust parity_plot.py to change the assessment criteria or visualization of the model performance.

Feel free to modify the code to suit your specific requirements and improve the model's performance.

