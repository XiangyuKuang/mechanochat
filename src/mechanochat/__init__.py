"""
MechanoChat
============================================
Mechanical Cell-Cell Communication Inference

Main Classes
----------------------------
SEM2
    2D SEM simulation class.
SEM3
    3D SEM simulation class.

Modules
----------------------------
pp  preprocessing
    mechanochat database loading and signal preprocessing
tl  tools
    Contact-based and Mechanical Cell-Cell Communication Inference
pl  plotting
    Visualization

"""

import os
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

import numba.cuda
if not numba.cuda.cudadrv.driver.USE_NV_BINDING:
    import warnings
    warnings.warn(
        "numba is not using the NVIDIA CUDA bindings. On driver 580+ / CUDA 13 this "
        "segfaults on the first CUDA call. Install `mechanochat[cuda11]` (or [cuda12]) "
        "and make sure NUMBA_CUDA_USE_NVIDIA_BINDING=1 is set before numba is imported.",
        RuntimeWarning,
    )

from .SEM import SEM2, SEM3
from . import preprocessing as pp
from . import plotting as pl
from . import tools as tl
__version__ = "0.1.0"

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp','tl','pl']})