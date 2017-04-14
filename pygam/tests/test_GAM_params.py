# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pygam import *


# TODO parameters get expanded: dtypes, n_splines, lam, fit_splines, fit linear
# TODO lines and splines
# TODO categorical dtypes get no fit linear even if fit linear TRUE
# TODO categorical dtypes get their own number of splines
# TODO can force continuous dtypes on categorical vars if wanted
