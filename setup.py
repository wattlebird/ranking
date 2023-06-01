from setuptools import setup, find_packages
import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name="rankit",
      version="0.3.3",
      packages=find_packages(exclude=['example']),
      install_requires=required,
      include_package_data=True,
      description="A simple ranking solution for matches.",
      long_description=open('Readme.rst').read(),
      author="Ronnie Wang",
      author_email="geniusxiaoguai@gmail.com",
      url="http://github.com/wattlebird/ranking",
      license="MIT",
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Unix',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      test_suite = 'nose.collector',
      )
