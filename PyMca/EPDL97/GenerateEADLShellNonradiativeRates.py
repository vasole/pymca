__doc__= "Generate specfiles with EADL97 shell transition probabilities" 
import os
import sys
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
    text += '#U01 directly extracted EADL97 nonradiative transition probabilities.\n'
    text += '#U02 EADL itself can be found at:\n'
    text += '#U03           http://www-nds.iaea.org/epdl97/libsall.htm\n'
    text += '#U04 The code used to generate this file has been:\n'
    text += '#U05 %s\n' % os.path.basename(__file__)
    text += '#U06\n'
    text += '\n'
    return text
shellList = EADLParser.getBaseShellList()
workingShells = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5'] 
for shell in workingShells:
    fname = "EADL97_%sShellNonradiativeRates.dat" % shell[0]
    print("fname = %s" % fname)
    if shell in ['K', 'L1', 'M1']:
        if os.path.exists(fname):
            os.remove(fname)
        nscan = 0
        outfile = open(fname, 'wb')
        tmpText = getHeader(fname)
        if sys.version < '3.0':
            outfile.write(tmpText)
        else:
            outfile.write(tmpText.encode('UTF-8'))            
    nscan += 1
    for i in range(1,101):
        print("Z = %d, Element = %s" % (i, Elements[i-1]))
        element = Elements[i-1]
        ddict = {}
        for key0 in shellList:
            tmpKey = key0.split()[0]
            if tmpKey in workingShells:
                if workingShells.index(tmpKey) <= workingShells.index(shell):
                    continue                    
            for key1 in shellList:
                tmpKey = key1.split()[0]
                if tmpKey in workingShells:
                    if workingShells.index(tmpKey) <= workingShells.index(shell):
                        continue                    
                key = "%s-%s%s" % (shell, key0.split()[0], key1.split()[0])
                if shell in [key0.split()[0], key1.split()[0]]:
                    continue
                ddict[key] = [0.0, 0.0]
        try:
            ddict = EADLParser.getNonradiativeTransitionProbabilities(\
                                    Elements.index(element)+1,
                                    shell=shell)
            print("%s Shell nonradiative emission probabilities " % shell)
        except IOError:
            #This happens when reading elements not presenting the transitions
            pass
            #continue
        if i == 1:
            #generate the labels
            nTransitions = 0
            tmpText = '#L Z  TOTAL'
            for key0 in workingShells:
                tmpKey = key0.split()[0]
                if tmpKey in workingShells:
                    if workingShells.index(tmpKey) <= workingShells.index(shell):
                        continue                    
                for key1 in shellList:
                    tmpKey = key1.split()[0]
                    if tmpKey in workingShells:
                        if workingShells.index(tmpKey) <= workingShells.index(shell):
                            continue
                    key = "%s-%s%s" % (shell, key0.split()[0], key1.split()[0])
                    tmpText += '  %s' % (key)
                    nTransitions += 1
            text  = '#S %d %s-Shell nonradiative rates\n' % (nscan, shell)
            text += '#N %d\n' % (2 + nTransitions)
            text += tmpText + '\n'
        else:
            text = ''
        # this loop calculates the totals, because it cannot be deduced from the subset
        # transitions written in the file
        total = 0.0
        for key0 in shellList:
            tmpKey = key0.split()[0]
            if tmpKey in workingShells:
                if workingShells.index(tmpKey) <= workingShells.index(shell):
                    continue                    
            for key1 in shellList:
                tmpKey = key1.split()[0]
                if tmpKey in workingShells:
                    if workingShells.index(tmpKey) <= workingShells.index(shell):
                        continue                    
                key = "%s-%s%s" % (shell, key0.split()[0], key1.split()[0])
                total += ddict.get(key, [0.0, 0.0])[0]
        text += '%d  %.7E' % (i, total)
        for key0 in workingShells:
            tmpKey = key0.split()[0]
            if tmpKey in workingShells:
                if workingShells.index(tmpKey) <= workingShells.index(shell):
                    continue                    
            for key1 in shellList:
                tmpKey = key1.split()[0]
                if tmpKey in workingShells:
                    if workingShells.index(tmpKey) <= workingShells.index(shell):
                        continue                    
                key = "%s-%s%s" % (shell, key0.split()[0], key1.split()[0])
                text += '  %.7E' % ddict.get(key, [0.0, 0.0])[0]
        text += '\n'
        if sys.version < '3.0':
            outfile.write(text)
        else:
            outfile.write(text.encode('UTF-8'))
    if sys.version < '3.0':
        outfile.write('\n')
    else:
        outfile.write('\n'.encode('UTF-8'))
if sys.version < '3.0':
    outfile.write('\n')
else:
    outfile.write('\n'.encode('UTF-8'))
outfile.close()
