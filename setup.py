import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

DISTNAME = 'smoothconco'
DESCRIPTION = 'Smoothed Concomitant Lasso solver'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'ndiayeeugene@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/EugeneNdiaye/smoothed_concomitant_lasso.git'
URL = 'https://github.com/EugeneNdiaye/smoothed_concomitant_lasso.git'
VERSION = None

setup(name='smoothconco',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['smoothconco'],
      ext_modules=cythonize("smoothconco/*.pyx"),
      include_dirs=[np.get_include()]
      )
