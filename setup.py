from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["cd_smoothed_concomitant.pyx",
                           "cd_smoothed_concomitant_screening.pyx"]),
    include_dirs=[np.get_include()]
)
