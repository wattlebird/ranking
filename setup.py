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
  Extension("rankit.util.fast_converter",
            ["rankit/util/fast_converter.pyx"],
            include_dirs=[numpy.get_include()],
            libraries=[]
  ),
  Extension("rankit.manager.fast_list_matrix",
            ["rankit/manager/fast_list_matrix.pyx"],
            include_dirs=[numpy.get_include()],
            libraries=[]
  ),
  Extension("rankit.Table.convert",
            ["rankit/Table/convert.pyx"],
            include_dirs=[numpy.get_include()]
  ),
  Extension("rankit.Ranker.matrix_build",
            ["rankit/Ranker/matrix_build.pyx"],
            include_dirs=[numpy.get_include()]
  )
]

setup(name="rankit",
      version="0.1.1",
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
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'],
      test_suite = 'nose.collector'
      )
