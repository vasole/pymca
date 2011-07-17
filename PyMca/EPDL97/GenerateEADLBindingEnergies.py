__doc__= "Generate specfile with EADL97 binding energies in keV" 
import os
import sys
import EADLSubshells
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

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = "EADL97_BindingEnergies.dat"

def getHeader(filename):
    text  = '#F %s\n' % filename
    text += '#U00 This file is a conversion to specfile format of \n'
    text += '#U01 directly extracted EADL97 Binding energies.\n'
    text += '#U02 EADL itself can be found at:\n'
    text += '#U03           http://www-nds.iaea.org/epdl97/libsall.htm\n'
    text += '#U04 The code used to generate this file has been:\n'
    text += '#U05 %s\n' % os.path.basename(__file__)
    text += '#U06\n'
    text += '\n'
    return text

if os.path.exists(fname):
    os.remove(fname)

outfile = open(fname, 'wb')
outfile.write(getHeader(fname))


shells = EADLParser.getBaseShellList()
LONG_LABEL = True
for i in range(1,101):
    print(i, Elements[i-1])
    if i == 1:
        text  = '#S 1 Binding energies in keV\n'
        label_text = ""
        n = 0
        for label in shells:
            if LONG_LABEL:
                label_text += "  "+label.replace(' ','')
            else:
                label_text += '  %s' % label.replace(' ','').split("(")[0]
            n += 1
        text += '#N %d\n' % n
        text += '#L Z' + label_text 
        text += '\n'
        outfile.write(text)
    text = "%d" % i 
    ddict = EADLParser.getBindingEnergies(i)
    for shell in shells:
        text += '  %.7E' % (ddict[shell] * 1000.)
    text += '\n'
    outfile.write(text)
outfile.write("\n")
outfile.close()
