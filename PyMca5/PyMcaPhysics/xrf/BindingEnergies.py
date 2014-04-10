#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__revision__ = "$Revision: 1.2 $"

import os
import numpy
try:
    from PyMca5 import specfile
except ImportError:
    print("BindingEnergies.py is importing specfile from local directory")
    import specfile

# PyMcaDataDir is created at installation time in setup.py
from PyMca5 import PyMcaDataDir

filename = "BindingEnergies.dat"
dirname = PyMcaDataDir.PYMCA_DATA_DIR
inputfile = os.path.join(dirname, filename)
if not os.path.exists(inputfile):
    dirname = os.path.dirname(dirname)
    inputfile = os.path.join(dirname, filename)
    if not os.path.exists(inputfile):
        if dirname.lower().endswith(".zip"):
            dirname = os.path.dirname(dirname)
            inputfile = os.path.join(dirname, filename)
    if not os.path.exists(inputfile):
        print("Cannot find inputfile ", inputfile)
        raise IOError("Cannot find BindingEnergies.dat file")

sf = specfile.Specfile(os.path.join(dirname, filename))
ElementShells = sf[0].alllabels()
ElementBinding = numpy.transpose(sf[0].data()).tolist()
sf = None

Elements = ['H', 'He',
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
            'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
            'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
            'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
            'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
            'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
            'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
            'Bh', 'Hs', 'Mt']


def main():
    import sys
    if len(sys.argv) > 1:
        ele = sys.argv[1]
        if ele in Elements:
            z = Elements.index(ele) + 1
            for shell in ElementShells:
                i = ElementShells.index(shell)
                if ElementBinding[z - 1][i] > 0.0:
                    print(shell, ElementBinding[z - 1][i])
            sys.exit()
    print("Usage:")
    print("python BindingEnergies.py [element]")

if __name__ == "__main__":
    main()
