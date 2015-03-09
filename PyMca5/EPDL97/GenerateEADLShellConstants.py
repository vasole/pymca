#/*##########################################################################
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__= "Generate specfiles with EADL97 shell constans"
import os
import EADLParser

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

def getHeader(filename):
    text  = '#F %s\n' % filename
    text += '#U00 This file is a conversion to specfile format of \n'
    text += '#U01 directly extracted EADL97 Shell constants.\n'
    text += '#U02 EADL itself can be found at:\n'
    text += '#U03           http://www-nds.iaea.org/epdl97/libsall.htm\n'
    text += '#U04 The code used to generate this file has been:\n'
    text += '#U05 %s\n' % os.path.basename(__file__)
    text += '#U06\n'
    text += '\n'
    return text

if __name__ == "__main__":
    #K Shell
    fname = "EADL97_KShellConstants.dat"
    if os.path.exists(fname):
        os.remove(fname)
    outfile = open(fname, 'wb')
    outfile.write(getHeader(fname))
    for i in range(1,101):
    #for i in range(82,83):
        print("%d %s" % (i, Elements[i-1]))
        if i == 1:
            text  = '#S 1 K Shell Fluorescence Yields\n'
            label_text = ""
            n = 1
            label_text += '  omegaK'
            n += 1
            text += '#N %d\n' % n
            text += '#L Z' + label_text
            text += '\n'
            outfile.write(text)
        text = "%d" % i
        ddict = EADLParser.getFluorescenceYields(i)
        value = ddict.get('K (1s1/2)', 0.0)
        text += '  %.4E' % (value)
        text += '\n'
        outfile.write(text)
    outfile.write("\n")
    outfile.close()

    #L Shell
    fname = "EADL97_LShellConstants.dat"
    if os.path.exists(fname):
        os.remove(fname)
    outfile = open(fname, 'wb')
    outfile.write(getHeader(fname))
    shell_list =  ['L1 (2s1/2)', 'L2 (2p1/2)', 'L3 (2p3/2)']
    for nshell in range(1,4):
        shell = shell_list[nshell-1]
        for i in range(1,101):
        #for i in range(82,83):
            print("%d %s" % (i, Elements[i-1]))
            if i == 1:
                text  = '#S %s %s x-ray data\n' % (shell[1], shell[0:2])
                label_text = ""
                n = 1
                if nshell == 1:
                    label_text += '  f12  f13  omegaL1'
                    n += 3
                elif nshell == 2:
                    label_text += '  f23  omegaL2'
                    n += 2
                else:
                    label_text += '  omegaL3'
                    n += 1
                text += '#N %d\n' % n
                text += '#L Z' + label_text
                text += '\n'
                outfile.write(text)
            text = "%d" % i
            ddict = EADLParser.getFluorescenceYields(i)
            ddict.update(EADLParser.getLShellCosterKronigYields(i))
            omega = ddict.get(shell, 0.0)
            if nshell == 1:
                f12 = ddict.get('f12', 0.0)
                f13 = ddict.get('f13', 0.0)
                text += '  %.4E  %.4E  %.4E\n' % (f12, f13, omega)
            elif nshell == 2:
                f23 = ddict.get('f23', 0.0)
                text += '  %.4E  %.4E\n' % (f23, omega)
            elif nshell == 3:
                text += '  %.4E\n' % (omega)
            outfile.write(text)
        outfile.write("\n")
    outfile.close()

    #M Shell
    fname = "EADL97_MShellConstants.dat"
    if os.path.exists(fname):
        os.remove(fname)
    outfile = open(fname, 'wb')
    outfile.write(getHeader(fname))
    shell_list =  ['M1 (3s1/2)',
                   'M2 (3p1/2)',
                   'M3 (3p3/2)',
                   'M4 (3d3/2)',
                   'M5 (3d5/2)']
    for nshell in range(1,6):
        shell = shell_list[nshell-1]
        for i in range(1,101):
        #for i in range(82,83):
            print("%d %s" % (i, Elements[i-1]))
            if i == 1:
                text  = '#S %s %s x-ray data\n' % (shell[1], shell[0:2])
                label_text = ""
                n = 1
                if nshell == 1:
                    label_text += '  f12  f13  f14  f15  omegaM1'
                    n += 5
                elif nshell == 2:
                    label_text += '  f23  f24  f25  omegaM2'
                    n += 4
                elif nshell == 3:
                    label_text += '  f34  f35  omegaM3'
                    n += 3
                elif nshell == 4:
                    label_text += '  f45  omegaM4'
                    n += 2
                else:
                    label_text += '  omegaM5'
                    n += 1
                text += '#N %d\n' % n
                text += '#L Z' + label_text
                text += '\n'
                outfile.write(text)
            text = "%d" % i
            ddict = EADLParser.getFluorescenceYields(i)
            ddict.update(EADLParser.getMShellCosterKronigYields(i))
            omega = ddict.get(shell, 0.0)
            if nshell == 1:
                f12 = ddict.get('f12', 0.0)
                f13 = ddict.get('f13', 0.0)
                f14 = ddict.get('f14', 0.0)
                f15 = ddict.get('f15', 0.0)
                text += '  %.4E  %.4E  %.4E  %.4E  %.4E\n' % (f12, f13, f14, f15, omega)
            elif nshell == 2:
                f23 = ddict.get('f23', 0.0)
                f24 = ddict.get('f24', 0.0)
                f25 = ddict.get('f25', 0.0)
                text += '  %.4E  %.4E  %.4E  %.4E\n' % (f23, f24, f25, omega)
            elif nshell == 3:
                f34 = ddict.get('f34', 0.0)
                f35 = ddict.get('f35', 0.0)
                text += '  %.4E  %.4E  %.4E\n' % (f34, f35, omega)
            elif nshell == 4:
                f45 = ddict.get('f45', 0.0)
                text += '  %.4E  %.4E\n' % (f45, omega)
            else:
                text += '  %.4E\n' % (omega)
            outfile.write(text)
        outfile.write("\n")
    outfile.close()
