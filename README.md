# ActiveLearning_OER
Active learning based model to sufficiently navigate OER catalyst

1. GPRdataspace.py : generates all the features for the surface motif. This represents the total dataspace that requires prediciton of energies
2. possibleFp.py : due to the periodic boundary condition of plane-wave basis set, a DFT-calculated dataspace is generated. The candidates for DFT-calcalculation is selected within this dataspace.
3. helperMethods.py : contains simple mathematical functions that help featurization process
4. motif_to_feature.py : converts the surface motif of the adsorption site to feature vector
5. mygaussian.py : Our active learning model. Contains gaussian process regression model and acquistion function for selecting the next calculated points
6. sum_element.py : preprocessing of data for activity calculation. For each feature, the number of atoms of three elements are summed.
7. activity_plot.py : based on the Boltzmann distribution of reaction kinetics, the activity of ternary catalyst is calculated.
8. parity_plot.py : assessment of model with final dataset 
