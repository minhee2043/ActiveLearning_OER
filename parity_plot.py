import sys
import os
from scipy.stats import norm
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pandas as pd
from pandas import DataFrame
from sklearn.gaussian_process.kernels import *

config = {
    # Font settings
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.titlepad': 5,
    'axes.linewidth': 0.75,
    
    
    'figure.figsize': (2, 2),
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,
    'figure.dpi': 300,
    
   
    'legend.frameon': False,
    'lines.linewidth': 0.75,
    'lines.markersize': 3,
    'xtick.major.size': 2.5,
    'xtick.major.pad': 2.5,
    'ytick.major.size': 2.5,
    'ytick.major.pad': 2.5,
}
plt.rcParams.update(config)

# Utility function to validate length scale parameters
def _check_length_scale(X, length_scale):
    """
    Validates the length scale parameter for the kernel
    """
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale

# Custom RBF kernel for integer-valued variables
class RBF_int(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    Modified RBF kernel for integer-valued variables, based on the paper
    "Dealing with categorical and integer-valued variables in Bayesian optimization with Gaussian Processes"
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
    
    # ... [kernel implementation details] ...

# Load and prepare data
x_all, y_all = [], []
with open('DFT_diffslab.csv', 'r') as handle:
    for line in handle.readlines()[0:]:
        # Parse 15 features and energy value from each line
        *features, energy = line[:].split(',')[:16]
        x_all.append(features)
        y_all.append(float(energy))


x_all = np.array(x_all)
X_all = np.array(x_all, dtype=int)
y_all = np.array(y_all)


x_train, x_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42
)

# Define the GPR kernel
# Combines a ConstantKernel with the custom RBF kernel for integer variables
kernel = gp.kernels.ConstantKernel(5.0, (1e-1, 1e3)) * RBF_int(
    length_scale=0.25*np.ones((15,)),
    length_scale_bounds=(1.0e-1, 1.0e3)
)

# Create and train the GPR model
model = gp.GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    alpha=0.1,
    normalize_y=True
)
model.fit(x_train, y_train)


y_pred = model.predict(x_test, return_std=False)
y_pred2 = model.predict(x_train, return_std=False)

# Calculate error metrics
MAE = round(mean_absolute_error(y_test, y_pred), 3)
RMSE = round(mean_squared_error(y_test, y_pred)**0.5, 3)

# Create the prediction vs actual plot
plt.scatter(y_train, y_pred2, label='train', s=4)
plt.scatter(y_test, y_pred, label='test', s=4)
plt.xlim(3.0, 3.8)
plt.ylim(3.0, 3.8)


x = np.linspace(3.0, 3.8, 10)
plt.plot([2.9, 3.9], [2.9, 3.9], '--', marker='o', color='black')
plt.fill_between(x, x-0.05, x+0.05, alpha=0.2)


plt.xlabel('$\Delta E_{DFT}$'+' [eV]')
plt.ylabel('$\Delta E_{pred}$'+' [eV]')
plt.text(3.04, 3.72, f'MAE(test set) = {MAE}', color='black')
plt.text(3.04, 3.64, f'RMSE(test set) = {RMSE}', color='black')
plt.tick_params(axis='both', which='major')
