"""
GAM toolkit
"""
from importlib.metadata import version, PackageNotFoundError

from pygam.pygam import GAM
from pygam.pygam import LinearGAM
from pygam.pygam import LogisticGAM
from pygam.pygam import GammaGAM
from pygam.pygam import PoissonGAM
from pygam.pygam import InvGaussGAM
from pygam.pygam import ExpectileGAM

from pygam.terms import l
from pygam.terms import s
from pygam.terms import f
from pygam.terms import te
from pygam.terms import intercept

__all__ = [
    'GAM',
    'LinearGAM',
    'LogisticGAM',
    'GammaGAM',
    'PoissonGAM',
    'InvGaussGAM',
    'ExpectileGAM',
    'l',
    's',
    'f',
    'te',
    'intercept',
]


__version__ = "0.0.0"  # placeholder for dynamic versioning
try:
    __version__ = version("pygam")
except PackageNotFoundError:
    # package is not installed
    pass
