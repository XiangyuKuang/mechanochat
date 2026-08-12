"""
MechanoChat
============================================
Mechanical Cell-Cell Communication Inference

Main Classes
----------------------------
SEM2
    Primary 2D active SEM simulation class.
SEM3
    3D SEM simulation class.

Modules
----------------------------
pp  preprocessing
    LR database loading and signal preprocessing
tl  tools
    Contact-based and Mechanical Cell-Cell Communication Inference
pl  plotting
    Visualization

"""

import os
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

from .SEM import SEM, SEM1, SEM2, SEM3, cellshape_GT
from . import preprocessing as pp
from . import plotting as pl
from . import tools as tl
from . import temp
__version__ = "0.0.1"

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp','tl','pl']})