import glob
import os
import platform
import subprocess
import sys

from setuptools import Command, Extension, setup, find_packages
from setuptools.command.test import test as TestCommand

import pip
import Cython
from Cython.Build import cythonize
import numpy


try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements


parsed_requirements = parse_requirements(
    'requirements/prod.txt',
    session='workaround'
)

parsed_test_requirements = parse_requirements(
    'requirements/test.txt',
    session='workaround'
)

parsed_setup_requirements = parse_requirements(
    'requirements/setup.txt',
    session='workaround'
)

parsed_requirements = list(parsed_requirements)
test_requirements = list(parsed_test_requirements)
setup_requirements = list(parsed_setup_requirements)
requirements = [str(ir.requirement) for ir in parsed_requirements]
test_requirements = [str(tr.requirement) for tr in parsed_test_requirements]
setup_requirements = [str(tr.requirement) for tr in parsed_setup_requirements]


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()



def define_extensions(cythonize=False):

    compile_args = ['-fopenmp',
                    '-ffast-math']

    # There are problems with illegal ASM instructions
    # when using the Anaconda distribution (at least on OSX).
    # This could be because Anaconda uses its own assembler?
    # To work around this we do not add -march=native if we
    # know we're dealing with Anaconda
    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')

    if cythonize:
        sinr_cooc = "sinr/cooccurrence_cython.pyx"
        nfm = "sinr/nfm.pyx"
    else:
        sinr_cooc = "sinr/cooccurrence_cython.cpp"
        nfm = "sinr/nfm.cpp"

    return [Extension("sinr.cooccurrence_cython", [sinr_cooc],
                      language='C++',
                      libraries=["stdc++"],
                      extra_link_args=compile_args,
		      include_dirs=[numpy.get_include()],
                      extra_compile_args=compile_args),
            Extension("sinr.nfm", [nfm],
                      language='C++',
                      libraries=["stdc++"],
		      include_dirs=[numpy.get_include()],
                      extra_link_args=compile_args,
                      extra_compile_args=compile_args)
            ]


def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support.
    """

    # For macports and homebrew
    patterns = ['/opt/local/bin/gcc-mp-[0-9].[0-9]',
                '/opt/local/bin/gcc-mp-[0-9]',
                '/usr/local/bin/gcc-[0-9].[0-9]',
                '/usr/local/bin/gcc-[0-9]']

    if 'darwin' in platform.platform().lower():

        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ["CC"] = gcc

        else:
            raise Exception('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        import Cython
        from Cython.Build import cythonize

        cythonize(define_extensions(cythonize=True))


class Clean(Command):
    """
    Clean build files.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, '*.egg-info')])
        subprocess.call(['find', pth, '-name', '*.pyc', '-type', 'f', '-delete'])
        subprocess.call(['rm', os.path.join(pth, 'sinr', 'cooccurrence_cython.so')])


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['tests/']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)
 

setup(
    name='sinr',
    author='Thibault Prouteau',
    author_email='thibault.prouteau@univ-lemans.fr',
    python_requires='>=3.6',
    version='0.1.0',
    description=('Python implementation of graph community '
                 'based word embeddings (SINr)'),
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
	entry_points={
        'console_scripts': [
            'sinr=sinr.cli:main',
        ],
	},
    packages=find_packages(),
    setup_requires =setup_requirements,
    install_requires=requirements,
	tests_require=test_requirements,
    cmdclass={'test': PyTest, 'cythonize': Cythonize, 'clean': Clean},
    url='https://git-lium.univ-lemans.fr/tprouteau/sinr_embeddings',
    #download_url='',
    license='Apache 2.0',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: Apache Software License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    ext_modules=cythonize(define_extensions(True)),
	zip_safe=False,
)
