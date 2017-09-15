"""
GAM toolkit
"""

from __future__ import absolute_import

from pygam.pygam import GAM
from pygam.pygam import LinearGAM
from pygam.pygam import LogisticGAM
from pygam.pygam import GammaGAM
from pygam.pygam import PoissonGAM
from pygam.pygam import InvGaussGAM

__all__ = ['GAM', 'LinearGAM', 'LogisticGAM', 'GammaGAM', 'PoissonGAM',
           'InvGaussGAM']

__version__ = '0.3.0'
