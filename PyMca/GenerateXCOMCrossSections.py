__doc__= "Generate specfile from XCOM generated files"
import sys
import os
import numpy
from PyMca import Elements

if len(sys.argv) < 3:
    print("Usage:")
    print("python GenerateXCOMTotalCrossSections SPEC_output_filename Barns_flag")
    sys.exit(0)

def getHeader(filename):
    text  = '#F %s\n' % filename
    text += '#U00 This file is a direct conversion to specfile format of \n'
    text += '#U01 the XCOM selected-arrays output.\n'
    text += '#U02 \n'
    text += '#U03 XCOM itself can be found at:\n'
    text += '#U04   http://www.nist.gov/pml/data/xcom/index.cfm\n'
    text += '\n'
    return text

fname = sys.argv[1]
if os.path.exists(fname):
    os.remove(fname)

if int(sys.argv[2]):
    BARNS = True
else:
    BARNS = False
print("BARNS = %s" % BARNS)
outfile = open(fname, 'wb')
outfile.write(getHeader(fname))

for i in range(1, 101):
    ele = Elements.getsymbol(i)
    print("i = %d element = %s" % (i, ele))
    # force data readout
    dataDict = Elements.getelementmassattcoef(ele)

    # pure XCOM data
    dataDict = Elements.Element[ele]['xcom']

    # energy (keV)
    energy = dataDict['energy']

    # coherent (cm2/g)
    cohe = dataDict['coherent']
    
    # incoherent
    incohe = dataDict['compton']
    
    # photoelectric
    photo = dataDict['photo']

    # photoelectric
    pair = dataDict['pair']

    # total
    total = dataDict['total']

    # convert to keV and cut at 500 keV not done for XCOM
    # indices = numpy.nonzero(energy<=500.)
    # energy = energy[indices]
    # photo  = photo[indices]
    # cohe   = cohe[indices]
    # incohe = incohe[indices]

    # I do not cut at 500 keV. I need to take the pair production
    total = photo + cohe + incohe + pair

    #now I am ready to write a Specfile
    text  = '#S %d %s\n' % (i, ele)
    text += '#N 5\n'
    labels = '#L PhotonEnergy[keV]'
    labels += '  Rayleigh(coherent)[barn/atom]' 
    labels += '  Compton(incoherent)[barn/atom]'
    labels += '  CoherentPlusIncoherent[barn/atom]'
    labels += '  Photoelectric[barn/atom]'
    labels += '  PairProduction[barn/atom]'
    labels += '  TotalCrossSection[barn/atom]\n'
    if not BARNS:
        labels = labels.replace("barn/atom", "cm2/g")
        factor = 1.0
    else:
        factor = Elements.Element[ele]['mass'] /(1.0E-24*AVOGADRO_NUMBER)
    text += labels
    if 0:
        fformat = "%g %g %g %g %g %g %g\n"
    else:
        fformat = "%.6E %.6E %.6E %.6E %.6E %.6E %.6E\n"
    outfile.write(text)
    for n in range(len(energy)):
        line = fformat % (energy[n],
                          cohe[n] * factor,
                          incohe[n] * factor,
                          (cohe[n] + incohe[n]) * factor,
                          photo[n] * factor,
                          pair[n] * factor,
                          total[n] * factor)
        outfile.write(line)
    outfile.write('\n')
outfile.close()
