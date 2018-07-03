import os

import numpy as np

from distutils.core import setup, Extension
from Cython.Distutils import build_ext


descr = 'Smooth Concomitant Lasso solver'

version = None
with open(os.path.join('smoothconco', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'smoothconco'
DESCRIPTION = descr
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'XXX'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'XXX'
VERSION = version
URL = 'XXX'

setup(name='smoothconco',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['smoothconco'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('smoothconco.cd_smoothed_concomitant',
                    sources=['smoothconco/cd_smoothed_concomitant.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('smoothconco.cd_smoothed_concomitant_screening',
                    sources=['smoothconco/cd_smoothed_concomitant_screening.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
                 ],
      )
