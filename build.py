import glob
import os
import platform
import subprocess
import sys

from setuptools import Command, Extension, Distribution
from setuptools.command.test import test as TestCommand
from Cython.Distutils.build_ext import new_build_ext as cython_build_ext

import Cython
from Cython.Build import cythonize
import numpy



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
        sinr_cooc = "sinr/text/cooccurrence_cython.pyx"
        #nfm = "sinr/nfm.pyx"
    else:
        sinr_cooc = "sinr/text/cooccurrence_cython.cpp"
        #nfm = "sinr/nfm.cpp"

    return [Extension("sinr.text.cooccurrence_cython", [sinr_cooc],
                      language='C++',
                      libraries=["stdc++"],
                      extra_link_args=compile_args,
		      include_dirs=[numpy.get_include()],
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


extension_modules = cythonize(define_extensions(True), language_level = "3")

distribution = Distribution({
    "ext_modules": extension_modules,
    "cmdclass" : {
        "build_ext": cython_build_ext,
    },
})

set_gcc()
distribution.run_command("build_ext")
build_ext_cmd = distribution.get_command_obj("build_ext")
build_ext_cmd.copy_extensions_to_source()
