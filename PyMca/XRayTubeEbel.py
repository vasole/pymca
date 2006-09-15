#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem to you.
#############################################################################*/
import Elements
import Numeric

DEBUG = 0

def continuumEbel(target, e0, e = None, window = None,
                  alphae = None, alphax = None,
                  transmission = None, targetthickness = None):
    """
    H. Ebel, X-Ray Spectrometry 28 (1999) 255-266
    Tube voltage from 5 to 50 kV
    Electron incident angle from 50 to 90 deg.
    X-Ray take off angle from 90 to 5 deg.
    def continuum(target, e, e0, window = None, mode=None)
    target = Tuple [Symbol, density, thickness] 
    e0 = Tube Voltage in kV     
    e  = Energy of interest
    window = Tube window [Formula, density, thickness] 
    """
    if type(target) == type([]):
        element = target[0]
        density = target[1]
        thickness = target[2]
    else:
        element   = target
        density   = Elements.Element[element]['density']
        thickness = 0.1
    if e is None:
        energy = Numeric.arange(e0 * 1.0)[1:]
    elif type(e) == type([]):
        energy = Numeric.array(e).astype(Numeric.Float)        
    elif type(e) == Numeric.ArrayType:
        energy = Numeric.array(e).astype(Numeric.Float)        
    else:
        energy = Numeric.array([e]).astype(Numeric.Float)

    if alphae is None:alphae = 75.0
    if alphax is None:alphax = 15.0
    if transmission is None:transmission = False

    sinalphae = Numeric.sin(Numeric.pi * alphae/180.)
    sinalphax = Numeric.sin(Numeric.pi * alphax/180.)
    sinfactor = sinalphae/sinalphax

    z = Elements.getz(element)
    const = 1.35e+09
    x = 1.109 - 0.00435*z + 0.00175 * e0


    #calculate intermediate constants from formulae (4) in Ebel's paper
    #eta in Ebel's paper
    m   = 0.1382 - 0.9211 / Numeric.sqrt(z)
    logz = Numeric.log(z)
    eta = 0.1904 - 0.2236 * logz + 0.1292 * pow(logz, 2) - 0.0149 * pow(logz, 3) 
    eta = eta * pow(e0, m) 

    #dephmax? in Ebel's paper
    p3 = 0.787e-05 * Numeric.sqrt(0.0135 * z) * pow(e0, 1.5) + \
         0.735e-06 * pow(e0, 2)
    rhozmax = (Elements.Element[element]['mass'] / z) * p3
    #print "max depth = ",2 * rhozmax

    #and finally we get rhoz 
    u0    = e0 / energy
    logu0 = Numeric.log(u0) 
    p1 = logu0 * (0.49269 - 1.09870 * eta + 0.78557 * pow(eta, 2))
    p2 = 0.70256 - 1.09865 * eta + 1.00460 * pow(eta, 2) + logu0
    rhoz = rhozmax * (p1 / p2)
    

    #the term dealing with the photoelectric absorption of the Bremsstrahlung
    tau = Numeric.array(Elements.getMaterialMassAttenuationCoefficients(element,
                                                          1.0,
                                                          energy)['photo'])

    if not transmission:
        rhelp = tau * 2.0 * rhoz * sinfactor
        if len(Numeric.nonzero(rhelp) <= 0.0):
            result = Numeric.zeros(Numeric.shape(rhelp), Numeric.Float)
            for i in range(len(rhelp)):
                if rhelp[i] > 0.0:
                    result [i] = const * z * pow(u0[i] - 1.0, x) * \
                         (1.0 - Numeric.exp(-rhelp[i])) / rhelp[i]
        else:
            result = const * z * pow(u0 - 1.0, x) * \
                 (1.0 - Numeric.exp(-rhelp)) / rhelp 

        #the term dealing with absorption in tube's window
        if window is not None:
            w = Elements.getMaterialTransmission(window[0], 1.0, energy,
                                                 density = window[1], thickness = window[2])['transmission']
            result = result * w
        return result
    #transmission case
    if targetthickness is None:
        #d = Elements.Element[target]['density']
        d = density
        ttarget = 2 * rhozmax
        print "WARNING target thickness assumed equal to maximum depth of %f cm" % (ttarget/d)
    else:
        #ttarget = targetthickness * Elements.Element[target]['density']
        ttarget = targetthickness * density
    generationdepth = min(ttarget, 2 * rhozmax)
    rhelp = tau * 2.0 * rhoz * sinfactor
    if len(Numeric.nonzero(rhelp) <= 0.0):
        result = Numeric.zeros(Numeric.shape(rhelp), Numeric.Float)
        for i in range(len(rhelp)):
            if rhelp[i] > 0.0:
                result [i] = const * z * pow(u0[i] - 1.0, x) * \
                     (Numeric.exp(-tau[i] *(ttarget-2.0*rhoz[i])/sinalphax)-Numeric.exp(-tau[i]*ttarget/sinalphax)) / rhelp[i]
    else:
        result = const * z * pow(u0 - 1.0, x) * \
             (Numeric.exp(-tau *(ttarget-2.0*rhoz)/sinalphax)-Numeric.exp(-tau*ttarget/sinalphax)) / rhelp
    #the term dealing with absorption in tube's window
    if window is not None:
        w = Elements.getMaterialTransmission(window[0], 1.0, energy,
                                             density = window[1], thickness = window[2]/sinalphax)['transmission']
        result = result * w
    return result

def characteristicEbel(target, e0, window = None,
                       alphae = None, alphax = None,
                       transmission = None, targetthickness = None):
    """
    H. Ebel, X-Ray Spectrometry 28 (1999) 255-266
    Tube voltage from 5 to 50 kV
    Electron incident angle from 50 to 90 deg.
    X-Ray take off angle from 90 to 5 deg.
    def continuum(target, e, e0, window = None, mode=None)
    target = Tuple [Symbol, density, thickness] 
    e0 = Tube Voltage in kV     
    e  = Energy of interest
    window = Tube window [Formula, density, thickness] 
    """
    if type(target) == type([]):
        element = target[0]
        density = target[1]
        thickness = target[2]
        if targetthickness is None:
            targetthickness = target[2] 
    else:
        element   = target
        density   = Elements.Element[element]['density']
        thickness = 0.1

    if alphae is None:alphae = 75.0
    if alphax is None:alphax = 15.0
    if transmission is None:transmission = False

    sinalphae = Numeric.sin(Numeric.pi * alphae/180.)
    sinalphax = Numeric.sin(Numeric.pi * alphax/180.)
    sinfactor = sinalphae/sinalphax

    z = Elements.getz(element)
    const = 6.0e+13
    #K Shell
    energy = Elements.Element[element]['binding']['K']
    #get the energy of the characteristic lines
    lines = Elements._getUnfilteredElementDict(element, None, photoweights = True)

    if 0:
        #L shell lines will have to be entered directly by the user
        #L shell 
        lpeaks = []
        for label in lines['L xrays']:
            lpeaks.append([lines[label]['energy'],
                              lines[label]['rate'], 
                              element+' '+label])
        lfluo = Elements._filterPeaks(peaklist, ethreshold = 0.020,
                                            ithreshold = 0.001,
                                            nthreshold = 6,
                                            absoluteithreshold = False,
                                            keeptotalrate = True)
        lfluo.sort()
    peaklist = []
    rays = 'K xrays'
    if rays in lines.keys():
        #K shell
        for label in lines[rays]:
            peaklist.append([lines[label]['energy'],
                              lines[label]['rate'], 
                              element+' '+label])
        fl = Elements._filterPeaks(peaklist, ethreshold = 0.020,
                                        ithreshold = 0.001,
                                        nthreshold = 4,
                                        absoluteithreshold = False,
                                        keeptotalrate = True)
        
        fl.sort()
        if (energy > 0) and (e0 > energy):
            zk = 2.0
            bk = 0.35
        else:
            for i in range(len(fl)):
                fl[i][1] = 0.00
            return fl

    u0    = e0 / energy
    logu0 = Numeric.log(u0) 

    #stopping factor
    oneovers = (Numeric.sqrt(u0) * logu0 + 2 * (1.0 - Numeric.sqrt(u0)))
    oneovers = oneovers / (u0 * logu0 + 1.0 - u0)
    oneovers = 1.0 + 16.05 * Numeric.sqrt(0.0137*z/energy) * oneovers
    oneovers = (zk*bk/z) * (u0 * logu0 + 1.0 - u0) * oneovers

    #backscattering factor
    r = 1.0 - 0.0081517 * z + 3.613e-05 * z * z +\
        0.009583 * z * Numeric.exp(-u0) + 0.001141 * e0

    #Absorption correction
    #calculate intermediate constants from formulae (4) in Ebel's paper
    #eta in Ebel's paper
    m   = 0.1382 - 0.9211 / Numeric.sqrt(z)
    logz = Numeric.log(z)
    eta = 0.1904 - 0.2236 * logz + 0.1292 * pow(logz, 2) - 0.0149 * pow(logz, 3) 
    eta = eta * pow(e0, m) 

    #depmax? in Ebel's paper
    p3 = 0.787e-05 * Numeric.sqrt(0.0135 * z) * pow(e0, 1.5) + \
         0.735e-06 * pow(e0, 2)
    rhozmax = (Elements.Element[element]['mass'] / z) * p3

    #and finally we get rhoz 
    p1 = logu0 * (0.49269 - 1.09870 * eta + 0.78557 * pow(eta, 2))
    p2 = 0.70256 - 1.09865 * eta + 1.00460 * pow(eta, 2) + logu0
    rhoz = rhozmax * (p1 / p2)

    #the term dealing with the photoelectric absorption
    energylist = []
    for i in range(len(fl)):
        energylist.append(fl[i][0])
    tau = Numeric.array(Elements.getMaterialMassAttenuationCoefficients(element,
                                                              1.0,
                                                              energylist)['photo'])
    if not transmission:
        rhelp = tau * 2.0 * rhoz * sinfactor
        if window is not None:
            w = Elements.getMaterialTransmission(window[0], 1.0, energylist,
                                             density = window[1], thickness = window[2])['transmission']
        for i in range(len(fl)):
            if rhelp[i] > 0.0 :
                rhelp[i] = (1.0 - Numeric.exp(-rhelp[i])) / rhelp[i]
            else:
                rhelp[i] = 0.0 
            intensity = const * oneovers * r * Elements.getomegak(element) * rhelp[i]
            #the term dealing with absorption in tube's window
            if window is not None:
                intensity = intensity * w[i]
            fl[i][1] = intensity * fl[i][1]
        return fl

    #transmission case
    if targetthickness is None:
        d = density
        ttarget = 2 * rhozmax
        print "WARNING target thickness assumed equal to maximum depth of %f cm" % (ttarget/d)
    else:
        ttarget = targetthickness * density
    generationdepth = min(ttarget, 2 * rhozmax)
    rhelp = tau * 2.0 * rhoz * sinfactor
    if window is not None:
        w = Elements.getMaterialTransmission(window[0], 1.0, energylist,
                                         density = window[1],
                                        thickness = window[2]/sinalphax)['transmission']
        for i in range(len(fl)):
            if rhelp[i] > 0.0:
                rhelp[i] = (Numeric.exp(-tau[i] *(ttarget-2.0*rhoz)/sinalphax)-Numeric.exp(-tau[i]*ttarget/sinalphax)) / rhelp[i]
            else:
                rhelp[i] = 0.0
            intensity = const * oneovers * r * Elements.getomegak(element) * rhelp[i]
            if window is not None:
                    intensity = intensity * w[i]
            fl[i][1] = intensity * fl[i][1]
    return fl


def generateLists(target, e0, window = None,
                  alphae = None, alphax = None,
                  transmission = None, targetthickness=None):
    e0w = 1.0 * e0
    x1min =  1.4
    step1 =  0.2
    #x1min = 8.0
    #step1 =  0.15
    x2min = min(e0, 20.0)
    #step2 =  0.3
    step2 =  0.5
    x3min = e0w
    x1    = Numeric.arange(x1min, x2min+step1, step1)
    x2    = Numeric.arange(x2min+step1, x3min, step2)

    #get K shell characteristic lines and intensities
    fllines = characteristicEbel(target, e0, window,
                                 alphae=alphae, alphax=alphax,
                                 transmission = transmission,
                                 targetthickness=targetthickness)
    
    energy = Numeric.ones(len(x1) + len(x2)).astype(Numeric.Float)
    energy[0:len(x1)] = energy[0:len(x1)] * x1
    energy[len(x1):(len(x1)+len(x2))] = energy[len(x1):(len(x1)+len(x2))] * x2
    #energy[(len(x1)+len(x2)):] = energy[(len(x1)+len(x2)):] * x3
    #print "len(energy) = ",len(energy)
    energyweight = continuumEbel(target, e0, energy, window,
                                 alphae=alphae, alphax=alphax,
                                 transmission = transmission,
                                 targetthickness=targetthickness)
    energyweight[0:len(x1)] *= step1
    energyweight[len(x1):(len(x1)+len(x2))] *= step2
    finalenergy = Numeric.zeros(len(fllines)+len(energyweight), Numeric.Float)
    finalweight = Numeric.zeros(len(fllines)+len(energyweight), Numeric.Float)
    scatterflag = Numeric.zeros(len(fllines)+len(energyweight))
    finalenergy[len(fllines):] = energy[0:]
    finalweight[len(fllines):] = energyweight[0:]/1.0e7
    for i in range(len(fllines)):
        finalenergy[i] = fllines[i][0]
        finalweight[i] = fllines[i][1]/1.0e7
        scatterflag[i] = 1
    return finalenergy, finalweight, scatterflag
    
    
if __name__ == "__main__":
    import sys
    import getopt
    options = ''
    longoptions = ['target=', 'voltage=', 'wele=', 'window=', 'wthickness=','anglee=', 'anglex=',
                   'cfg=', 'deltae=', 'transmission=','tthickness=']
    opts, args = getopt.getopt(
        sys.argv[1:],
        options,
        longoptions)
    target  = 'Ag'
    voltage = 40
    wele    = 'Be'
    wthickness = 0.0125
    anglee     = 70
    anglex     = 50
    cfgfile = None
    transmission = None
    ttarget = None
    for opt,arg in opts:
        if opt in ('--target'):
            target = arg
        elif opt in ('--tthickness'):
            ttarget = float(arg)
        if opt in ('--cfg'):
            cfgfile = arg
        if opt in ('--voltage'):
            voltage = float(arg)
        if opt in ('--wthickness'):
            wthickness = float(arg)
        if opt in ('--wele','window'):
            wele = arg
        if opt in ('--transmission'):
            transmission=int(arg)
        if opt in ('--anglee', '--alphae'):
            anglee=float(arg)
        if opt in ('--anglex', '--alphax'):
            anglex=float(arg)
    try:
        e = Numeric.arange(voltage*10+1)[1:]/10
        y= continuumEbel(target, voltage, e,
                        [wele, Elements.Element[wele]['density'], wthickness],
                         alphae=anglee, alphax=anglex,
                         transmission=transmission, targetthickness=ttarget)
        fllines = characteristicEbel(target, voltage,
                        [wele, Elements.Element[wele]['density'], wthickness],
                         alphae=anglee, alphax=anglex,
                         transmission=transmission, targetthickness=ttarget)
        fsum = 0.0
        for l in fllines:
            print "%s %.4f %.3e" % (l[2],l[0],l[1])
            fsum += l[1]
        energy, weight, scatter =  generateLists(target, voltage,
                        [wele, Elements.Element[wele]['density'], wthickness],
                         alphae=anglee, alphax=anglex,
                         transmission=transmission, targetthickness=ttarget)

        f = open("Tube_%s_%.1f_%s_%.5f_ae%.1f_ax%.1f.txt" % (target,voltage,wele,wthickness,anglee,anglex),"w+")
        text = "energyweight="
        for i in range(len(energy)):
            if i == 0:
                text +=" %f" % weight[i]
            else:
                text +=", %f" % weight[i]
        text +="\n"
        f.write(text)
        text = "energy="
        for i in range(len(energy)):
            if i == 0:
                text +=" %f" % energy[i]
            else:
                text +=", %f" % energy[i]
        text +="\n"
        f.write(text)
        text = "energyflag="
        for i in range(len(energy)):
            if i == 0:
                text +=" %f" % 1
            else:
                text +=", %f" % 1
        text +="\n"
        f.write(text)
        text = "energyscatter="
        for i in range(len(energy)):
            if i == 0:
                text +=" %f" % scatter[i]
            else:
                text +=", %f" % scatter[i]
        text +="\n"
        f.write(text)
        f.close()
    except:
        print "Usage:"
        print "options = ",longoptions
        sys.exit(0)
        
    

