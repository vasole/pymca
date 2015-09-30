#!/usr/bin/env python
# /*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
__author__ = "V.A. Sole, T. Vincent - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

"""Setup script for the Object3DQhull module distribution."""

import glob
import os
import numpy

from distutils.extension import Extension
from distutils.core import setup


# TODO For '../../../third-party/qhull/src/*.c'
# *.o are stored outside build/ when building locally

def qhull_ext_modules(parent_name=None, package_path=None, use_cython=True):
    """Build list of Qhull wrapper Extensions.

    :param str parent_name: Name of parent module (or None).
    :param str package_path: Directory where the source of the module are.
    :param bool use_cython: Whether to use Cython or C source files.
    :return: List of py_modules and List of Extensions to build as a tuple.
    """
    # Only try to import from Cython when used
    if use_cython:
        from Cython.Build import cythonize

    name_prefix = '' if parent_name is None else (parent_name + '.')
    if package_path is None:
        path_prefix = ''
    else:
        path_prefix = package_path.strip('/') + '/'

    sources = []
    include_dirs = [path_prefix + '_qhull', numpy.get_include()]
    extra_compile_args = []
    extra_link_args = []

    ext_modules = []

    QHULL_CFLAGS = os.getenv("QHULL_CFLAGS")
    QHULL_LIBS = os.getenv("QHULL_LIBS")
    use_system_qhull = QHULL_CFLAGS and QHULL_LIBS
    if use_system_qhull:
        # Use user provided system qhull library
        extra_compile_args += [QHULL_CFLAGS]
        extra_link_args += [QHULL_LIBS]

        # WARNING: MUST be sync with qhull/user.h
        qhull64_macros = {'REALfloat': 0, 'qh_QHpointer': 1}  # As in debian 7
        qhull32_macros = None

        # cython .c files need to be regenerated during build
        cythonize_force = True
        assert use_cython
    else:
        # Compile qhull
        sources += glob.glob(path_prefix +
                             '../../../third-party/qhull/src/*.c')
        include_dirs += [path_prefix + '../../../third-party/qhull/src']

        qhull64_macros = {'REALfloat': 0, 'qh_QHpointer': 0}
        qhull32_macros = {'REALfloat': 1, 'qh_QHpointer': 0}

        cythonize_force = False

    if qhull32_macros is not None:
        # qhull for float32
        qhull32_sources = sources + [
            path_prefix + '_qhull/_qhull32' + ('.pyx' if use_cython else '.c')]
        qhull32_ext = Extension(name=name_prefix + '_qhull32',
                                sources=qhull32_sources,
                                define_macros=list(qhull32_macros.items()),
                                include_dirs=include_dirs,
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args,
                                language='c')
        if use_cython:
            qhull32_ext = cythonize([qhull32_ext],
                                    force=cythonize_force,
                                    compile_time_env=qhull32_macros)[0]
        ext_modules.append(qhull32_ext)

    if qhull64_macros is not None:
        # qhull for float64
        qhull64_sources = sources + [
            path_prefix + '_qhull/_qhull64' + ('.pyx' if use_cython else '.c')]
        qhull64_ext = Extension(name=name_prefix + '_qhull64',
                                sources=qhull64_sources,
                                define_macros=list(qhull64_macros.items()),
                                include_dirs=include_dirs,
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args,
                                language='c')
        if use_cython:
            qhull64_ext = cythonize([qhull64_ext],
                                    force=cythonize_force,
                                    compile_time_env=qhull64_macros)[0]
        ext_modules.append(qhull64_ext)

    # py_modules
    py_modules = [name_prefix + '__init__']

    return py_modules, ext_modules


# Only build C extensions
setup(name="Object3DQhull",
      version="1.0",
      description="Interface to Qhull library.",
      author="V.A. Sole - Software Group",
      author_email="sole@esrf.fr",
      url="http://www.esrf.fr",
      ext_modules=qhull_ext_modules()[1],
      )
