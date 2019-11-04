from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext

print(':setup.py:[DEBUG]', "numpy is at", numpy.get_include())

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("lic_internal", ["lic_internal.pyx"], include_dirs=[
                           numpy.get_include()])]
)
