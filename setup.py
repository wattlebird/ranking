from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

with open('requirements.txt') as f:
    required = f.read().splitlines()

ext_modules=[
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

setup(name="rankit",
      version="0.1.0.dev0",
      packages=find_packages(exclude=['example']),
      install_requires=required,
      include_package_data=True,
      description="A linear algebra based ranking solution",
      author="Ronnie Wang",
      author_email="geniusxiaoguai@gmail.com",
      url="http://github.com/wattlebird/ranking",
      license="MIT",
      use_2to3=True,
      ext_modules=cythonize(ext_modules),
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',],
      test_suite = 'nose.collector'
      )
