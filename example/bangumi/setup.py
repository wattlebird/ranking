from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_modules=[
    Extension("util",
              sources=["util.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"] # Unix-like specific
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
