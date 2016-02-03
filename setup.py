from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(name="rankit",
      version="0.1.0dev0",
      packages=find_packages(exclude=['example']),
      include_package_data=True,
      description="A linear algebra based ranking solution",
      author="Ronnie Wang",
      author_email="geniusxiaoguai@gmail.com",
      url="http://github.com/wattlebird/ranking",
      license="MIT",
      use_2to3=True,
      ext_modules=[
          Extension("rankit/util/fast_converter",
                    sources = ["rankit/util/fast_converter.c"],
                    include_dirs=[numpy.get_include()],
                    libraries=["m"]
          ),
          Extension("rankit/manager/fast_list_matrix",
                    sources = ["rankit/manager/fast_list_matrix.c"],
                    include_dirs=[numpy.get_include()],
                    libraries=["m"]
          )
      ],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',],
      )
