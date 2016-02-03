from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_modules=[
    Extension("example/util",
              sources=["example/util.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"] # Unix-like specific
    ),
    Extension("rankit/util/fast_converter",
              sources = ["rankit/util/fast_converter.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"]
    ),
    Extension("rankit/manager/fast_list_matrix",
              sources = ["rankit/manager/fast_list_matrix.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["m"]
    )
]

setup(
    ext_modules = cythonize(ext_modules)
)
