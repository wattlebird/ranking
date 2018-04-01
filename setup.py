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
  Extension("rankit.Ranker.matrix_build",
            ["rankit/Ranker/matrix_build.pyx"],
            include_dirs=[numpy.get_include()]
  )
]

setup(name="rankit",
      version="0.2",
      packages=find_packages(exclude=['example']),
      install_requires=required,
      include_package_data=True,
      description="A simple ranking solution for matches.",
      author="Ronnie Wang",
      author_email="geniusxiaoguai@gmail.com",
      url="http://github.com/wattlebird/ranking",
      license="MIT",
      ext_modules=cythonize(ext_modules),
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      test_suite = 'nose.collector'
      )
