import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import (
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Kernel,
    Hyperparameter,
    WhiteKernel
)
from scipy.spatial.distance import pdist, cdist, squareform

def _check_length_scale(X, length_scale):
    """
    Validate the length scale parameter for the RBF kernel.
    
    Args:
        X (np.ndarray): Input data matrix
        length_scale (float or np.ndarray): Length scale parameter(s)
    
    Returns:
        np.ndarray: Validated length scale parameter(s)
    
    Raises:
        ValueError: If length_scale dimensions don't match data dimensions
    """
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            f"Anisotropic kernel must have the same number of dimensions as data ({length_scale.shape[0]}!={X.shape[1]})"
        )
    return length_scale

class RBF_int(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    Radial Basis Function kernel for integer-valued variables.
    Implementation based on "Dealing with categorical and integer-valued variables 
    in Bayesian optimization with Gaussian Processes" by Garrido-Merchan & Hernandez-Lobato.
    """
    
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        """
        Initialize the RBF kernel for integer-valued variables.
        
        Args:
            length_scale (float or np.ndarray): Length scale parameter(s)
            length_scale_bounds (tuple): Bounds for length scale optimization
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        """Check if the kernel is anisotropic (different length scales per dimension)."""
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        """Define the length scale hyperparameter properties."""
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute the kernel matrix between X and Y.
        
        Args:
            X (np.ndarray): First set of input points
            Y (np.ndarray, optional): Second set of input points
            eval_gradient (bool): Whether to evaluate the gradient
            
        Returns:
            np.ndarray: Kernel matrix (and gradient if eval_gradient=True)
        """
        X = np.atleast_2d(X)
        X = np.around(X)  # Round to nearest integer
        length_scale = _check_length_scale(X, self.length_scale)
        
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            else:
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        return K

def load_and_preprocess_data(filename):
    """
    Load and preprocess data from CSV file.
    
    Args:
        filename (str): Path to CSV file
        
    Returns:
        tuple: X features array and y target array
    """
    x_data, y_data = [], []
    with open(filename, 'r') as handle:
        for line in handle.readlines():
            features = line.split(',')[:15]
            energy = float(line.split(',')[15])
            x_data.append(features)
            y_data.append(energy)
    
    X = np.array(x_data, dtype=int)
    y = np.array(y_data)
    return X, y

def create_gpr_model(n_features):
    """
    Create and configure the Gaussian Process Regression model.
    
    Args:
        n_features (int): Number of input features
        
    Returns:
        GaussianProcessRegressor: Configured GPR model
    """
    kernel = gp.kernels.ConstantKernel(1, (1e-1, 1e3)) * RBF_int(
        length_scale=0.2 * np.ones((n_features,)),
        length_scale_bounds=(1.0e-1, 1.0e3)
    )
    
    return gp.GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.05,
        normalize_y=True
    )

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    
    Args:
        model: Trained GPR model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        
    Returns:
        float: Mean Absolute Error
    """
    y_pred = model.predict(X_test, return_std=False)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

def generate_batch_suggestions(model1, model2, output_file='GPR_batch<num>.csv'):
    """
    Generate and process batch suggestions based on model predictions.
    'total_GPRspace.csv' refers to the csv made from GPRdataspace.py
    Args:
        model1: First trained *O GPR model
        model2: Second trained *OH GPR model
        output_file (str): Path to output CSV file
    """
    # Load validation data
    x_val = []
    multiplicities = []
    with open('total_GPRspace.csv', 'r') as handle: 
        for line in handle.readlines():
            features = line.split(',')[:15]
            mult = int(line.split(',')[15])
            x_val.append(features)
            multiplicities.append(mult)
    
    X_val = np.array(x_val, dtype=int)
    
    # Get predictions and uncertainties from both models
    y_val1, dev1 = model1.predict(X_val, return_std=True)
    y_val2, dev2 = model2.predict(X_val, return_std=True)
    
    # Calculate importance score
    diff_from_target = -abs((y_val1 - y_val2) - 5.3)  # Target difference of 5.3
    Z = diff_from_target / ((dev1 + dev2) / 2)
    uncertainty = diff_from_target * norm.cdf(Z) + dev1 * norm.pdf(Z)
    
    # Combine all data for output
    output = np.c_[X_val, multiplicities, y_val1, y_val2, y_val1-y_val2, dev1, dev2, uncertainty]
    np.savetxt(output_file, output, fmt=['%d']*16 + ['%.5f']*6, delimiter=',')
    
    # Sort and process for suggestions
    data = pd.read_csv(output_file, header=None)
    data.sort_values(data.columns[21], axis=0, ascending=[False], inplace=True)
    data.to_csv('GPR_batch<num>_arrange.csv', header=None, index=None,
                columns=list(range(16)))  # Keep only first 16 columns
    
    # Generate final suggestions
    generate_final_suggestions(
        'GPR_batch<num>_arrange.csv',
        'possibleFp.csv',
        'batch<num+1>_suggest.csv',
        'index_metal.csv',
        'batch<num+1>_metal.csv'
    )

def generate_final_suggestions(arranged_file, possible_file, suggest_file, 
                             index_file, metal_file, max_suggestions=30):
    """
    Generate final batch suggestions by comparing arranged results with possible fingerprints.
    
    Args:
        arranged_file (str): Path to arranged GPR results by EI
        possible_file (str): Path to possible fingerprints file - made from possibleFp.py
        suggest_file (str): Path to surfaces suggested by aquisition function file
        index_file (str): Path to metal index file - made from possibleFp.py
        metal_file (str): Path to output metal file - file that converted suggested surface feature to structure information
        max_suggestions (int): Maximum number of suggestions to generate
    """
    # Read input files
    with open(arranged_file, 'r') as t1, open(possible_file, 'r') as t2:
        arranged_lines = t1.readlines()
        possible_lines = t2.readlines()
    
    # Find matching lines and their indices
    row_nums = []
    suggestions_count = 0
    
    with open(suggest_file, 'w') as outfile:
        for line in arranged_lines:
            if line in possible_lines:
                outfile.write(line)
                row_nums.append(possible_lines.index(line))
                suggestions_count += 1
                if suggestions_count >= max_suggestions:
                    break
    
    # Generate metal suggestions
    with open(index_file, 'r') as f:
        index_to_metal = f.readlines()
    
    with open(metal_file, 'w') as fout:
        for num in row_nums:
            fout.write(index_to_metal[num])

def main():
    # Load and process O batch data
    X_all, y_all = load_and_preprocess_data('DFT_O_all.csv')
    x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25)
    
    # Train and evaluate first model
    model1 = create_gpr_model(n_features=15)
    model1.fit(x_train, y_train)
    mae1 = evaluate_model(model1, x_test, y_test)
    print(f"Model 1 MAE: {mae1}")
    
    # Load and process OH batch data
    X_all2, y_all2 = load_and_preprocess_data('DFT_OH_all.csv')
    x_train2, x_test2, y_train2, y_test2 = train_test_split(X_all2, y_all2, test_size=0.25)
    
    # Train and evaluate second model
    model2 = create_gpr_model(n_features=15)
    model2.fit(x_train2, y_train2)
    mae2 = evaluate_model(model2, x_test2, y_test2)
    print(f"Model 2 MAE: {mae2}")
    
    # Generate batch suggestions
    generate_batch_suggestions(model1, model2)

if __name__ == "__main__":
    main()
