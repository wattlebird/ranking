from setuptools import setup, find_packages
from setuptools.extension import Extension
import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

with open('requirements.txt') as f:
    required = f.read().splitlines()

def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext#

    # include_dirs adjusted: 
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)

setup(name="rankit",
      version="0.3.0",
      packages=find_packages(exclude=['example']),
      setup_requires=['numpy', 'Cython'],
      install_requires=required,
      include_package_data=True,
      description="A simple ranking solution for matches.",
      author="Ronnie Wang",
      author_email="geniusxiaoguai@gmail.com",
      url="http://github.com/wattlebird/ranking",
      license="MIT",
      cmdclass={'build_ext' : my_build_ext},
      ext_modules=[
            Extension("rankit.Ranker.matrix_build",
                        ["rankit/Ranker/matrix_build.pyx"]
            )
      ],
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
