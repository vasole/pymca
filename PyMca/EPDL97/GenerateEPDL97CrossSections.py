__doc__= "Generate specfile from all EPL97 cross sections in keV and barn" 
import os
import sys
import EADLSubshells
import EPDLParser
Elements = EPDLParser.Elements
AVOGADRO_NUMBER = EPDLParser.AVOGADRO_NUMBER
import numpy
log = numpy.log
exp = numpy.exp
getTotalCoherentCrossSection = EPDLParser.getTotalCoherentCrossSection
getTotalIncoherentCrossSection = EPDLParser.getTotalIncoherentCrossSection
getTotalPhotoelectricCrossSection = EPDLParser.getTotalPhotoelectricCrossSection
getPartialPhotoelectricCrossSection = EPDLParser.getPartialPhotoelectricCrossSection
getTotalPairCrossSection = EPDLParser.getTotalPairCrossSection
getTotalTripletCrossSection = EPDLParser.getTotalTripletCrossSection

if len(sys.argv) < 3:
    print "Usage:"
    print "python EPDL97GenerateCrossSections SPEC_output_filename barns_flag"
    sys.exit(0)

def getHeader(filename):
    text  = '#F %s\n' % filename
    text += '#U00 This file is a direct conversion to specfile format of \n'
    text += '#U01 the original EPDL97 photoelectric cross sections contained\n'
    text += '#U02 in the EPDL97.DAT file from the library.\n'
    text += '#U03 EPDL97 itself can be found at:\n'
    text += '#U04           http://www-nds.iaea.org/epdl97/libsall.htm\n'
    text += '#U05\n'
    text += '#U06 The program used to generate this file has been:\n'
    text += '#U07 %s\n' % os.path.basename(__file__)
    text += '\n'
    return text

fname = sys.argv[1]
if os.path.exists(fname):
    os.remove(fname)

if int(sys.argv[2]):
    BARNS = True
else:
    BARNS = False
print "BARNS = ", BARNS
outfile = open(fname, 'wb')
outfile.write(getHeader(fname))

shells = EADLSubshells.SHELL_LIST
bad_shells = ['L (', 'L23',
              'M (', 'M23', 'M45',
              'N (', 'N23', 'N45', 'N67',
              'O (', 'O23', 'O45', 'O67', 'O89',
              'P (', 'P23', 'P45', 'P67', 'P89', 'P101',
              'Q (', 'Q23', 'Q45', 'Q67']
LONG_LABELS = True
for i in range(1, 101):
#for i in range(82, 83):
    print "i = ", i, "element = ", Elements[i-1]
    #coherent
    energy_cohe, value_cohe, mode_cohe = getTotalCoherentCrossSection(i,
                                                            getmode=True)
    #incoherent
    energy_incohe, value_incohe, mode_incohe = getTotalIncoherentCrossSection(i,
                                                            getmode=True)

    #photoelectric
    energy_photo, value_photo, mode_photo = getTotalPhotoelectricCrossSection(i,
                                                            getmode=True)

    #check to see the energies:
    #for j in range(10):
    #    print energy_cohe[j], energy_incohe[j], energy_photo[j]

    #to select an appropriate energy grid as close as possible to the original
    #while keeping in mind the PyMca goals, I use the coherent energy grid till
    #the non-zero first value of the photoelectric cross section. At that point,
    #I use the photoelectric energy grid.
    energy = numpy.concatenate((energy_cohe[energy_cohe<energy_photo[0]],
                               energy_photo))

    #now perform a log-log interpolation when needed
    #lin-lin interpolation:
    #
    #              y0 (x1-x) + y1 (x-x0)
    #        y = -------------------------
    #                     x1 - x0
    #
    #log-log interpolation:
    #
    #                  log(y0) * log(x1/x) + log(y1) * log(x/x0)
    #        log(y) = ------------------------------------------
    #                                  log (x1/x0)
    #
    cohe    = numpy.zeros(len(energy), numpy.float)
    incohe  = numpy.zeros(len(energy), numpy.float)
    photo   = numpy.zeros(len(energy), numpy.float)
    total   = numpy.zeros(len(energy), numpy.float)    

    #get the partial photoelectric cross sections
    photo_dict = {}
    photo_label_list = []
    photo_long_label_list = []
    END = False
    for shell in EADLSubshells.SHELL_LIST:
        if shell[0:3] in bad_shells:
            continue
        if shell[0:4] in bad_shells:
            continue
        if shell[0] in ['P', 'Q']:
            continue
        photo_long_label_list.append(shell)
        actual_shell = shell.replace(' ','').split("(")[0]
        photo_label_list.append(actual_shell)
        photo_dict[actual_shell] = {}
        ene = energy * 1
        v   = photo * 0.0
        value = photo * 0.0
        if not END:
            try:
                ene, v, mode = getPartialPhotoelectricCrossSection(i, actual_shell,
                                                                getmode=True)
                #log-log interpolate in the final energy grid
                value = photo * 0.0
                for n in range(len(energy)):
                    x = energy[n]
                    if (x == ene[0]) and (energy[n+1] == x):
                        #avoid entering twice the absorption edges
                        continue
                    if x == ene[0]:
                        value[n]=v[0]
                        continue
                    if x < ene[0]:
                        continue
                    j1 = 0
                    while ene[j1] < x:
                        j1 += 1
                    j0 = j1 - 1
                    x0 = ene[j0]
                    x1 = ene[j1]
                    y0 = v[j0]
                    y1 = v[j1]
                    value[n] = exp((log(y0) * log(x1/x) + log(y1) * log(x/x0))/log(x1/x0))
            except IOError:
                END = True
                #print sys.exc_info()
                ene = energy * 1
                v   = photo * 0.0
        photo_dict[actual_shell]['read_energy'] = ene
        photo_dict[actual_shell]['read_value']  = v
        photo_dict[actual_shell]['value']  = value

    #coherent needs to interpolate
    indices = numpy.nonzero(energy_cohe<energy_photo[0]) 
    cohe[indices]  = value_cohe[indices]
    for n in range(len(indices),len(energy)):
        x = energy[n]
        j1 = len(indices)
        if energy_cohe[j1] == x:
            cohe[n] = value_cohe[j1]
            continue
        while energy_cohe[j1] < x:
            j1 += 1
        j0 = j1 - 1
        if j0 < 0:
            print x, energy_cohe[0]
            raise ValueError, "coherent"
        x0 = energy_cohe[j0]
        x1 = energy_cohe[j1]
        y0 = value_cohe[j0]
        y1 = value_cohe[j1]
        cohe[n] = exp((log(y0) * log(x1/x) + log(y1) * log(x/x0))/log(x1/x0))

    #compton needs to interpolate everything
    for n in range(len(energy)):
        x = energy[n]
        j1 = 0
        if energy_incohe[j1] == x:
            incohe[n] = value_incohe[j1]
            continue
        while energy_incohe[j1] < x:
            j1 += 1
        j0 = j1 - 1
        if j0 < 0:
            print x, energy_incohe[0]
            raise ValueError, "compton"
        x0 = energy_incohe[j0]
        x1 = energy_incohe[j1]
        y0 = value_incohe[j0]
        y1 = value_incohe[j1]
        incohe[n] = exp((log(y0) * log(x1/x) + log(y1) * log(x/x0))/log(x1/x0))

    #photoelectric does not need to interpolate anything
    j1 = 0
    for n in range(len(energy)):
        x = energy[n]
        if x < energy_photo[0]:
            continue
        j1 = 0
        photo[n] = value_photo[j1]
        j1 += 1
    photo[energy>=energy_photo[0]] = value_photo[:]


    #convert to keV and cut at 500 keV
    energy *= 1000.
    indices = numpy.nonzero(energy<=500.)
    energy = energy[indices]
    photo  = photo[indices]
    cohe   = cohe[indices]
    incohe = incohe[indices]

    #I cut at 500 keV, I do not need to take the pair production
    total = photo + cohe + incohe

    #now I am ready to write a Specfile
    ele = Elements[i-1]
    text  = '#S %d %s\n' % (i, ele)
    text += '#N %d\n' % (7+len(photo_label_list))
    labels = '#L PhotonEnergy[keV]'
    labels += '  Rayleigh(coherent)[barn/atom]' 
    labels += '  Compton(incoherent)[barn/atom]'
    labels += '  CoherentPlusIncoherent[barn/atom]'
    labels += '  Photoelectric[barn/atom]'
    if LONG_LABELS:
        for label in photo_long_label_list:
            labels += "  "+label.replace(" ","")+"[barn/atom]"
    else:
        for label in photo_label_list:
            labels += "  "+label+"[barn/atom]"
    labels += "  AllOthers[barn/atom]"
    labels += '  TotalCrossSection[barn/atom]\n'
    if not BARNS:
        labels = labels.replace("barn/atom", "cm2/g")
        factor = (1.0E-24*AVOGADRO_NUMBER)/EPDLParser.getAtomicWeights()[i-1]
    else:
        factor = 1.0
    text += labels
    if 0:
        fformat = "%g %g %g %g %g"
    else:
        fformat = "%.7E %.6E %.6E %.6E %.6E"
    outfile.write(text)
    cohe   *= factor
    incohe *= factor
    photo  *= factor
    total  *= factor
    for n in range(len(energy)):
        line = fformat % (energy[n],
                          cohe[n],
                          incohe[n],
                          cohe[n]+incohe[n],
                          photo[n])
        d = 0.0
        for l in photo_label_list:
            a = photo_dict[l]['value'][n] * factor
            line += " %.6E" % a
            d += a
        line += " %.6E %.6E\n" % (photo[n]-d, total[n])       
        outfile.write(line)
outfile.write('\n')
outfile.close()
