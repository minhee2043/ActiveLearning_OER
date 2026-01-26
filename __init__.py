"""
Active Learning for OER Catalyst Discovery

A Python package for efficient catalyst discovery using active learning
and Gaussian Process Regression with surface configuration fingerprints.

"""

__version__ = '1.0.0'
__author__ = 'Minhee Park, Hayun Jeon'
__email__ = 'minhee2043@snu.ac.kr'

# Import main classes for easy access
from active_learning.core.features import Slab
from active_learning.core.dataspace import DataspaceGenerator
from active_learning.core.gpr import ActiveLearner, RBF_int
from active_learning.core.dft_compatible import DFTCompatibleGenerator
from active_learning.analysis.activity import ActivityCalculator
from active_learning.analysis.processing import ElementCounter

# Import utilities
from active_learning.utils import helpers

__all__ = [
    # Core classes
    'Slab',
    'DataspaceGenerator',
    'ActiveLearner',
    'RBF_int',
    'DFTCompatibleGenerator',
    
    # Analysis
    'ActivityCalculator',
    'ElementCounter',
    
    # Utilities
    'helpers',
]
