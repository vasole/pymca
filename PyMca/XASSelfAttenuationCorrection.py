# -*- coding: utf-8 -*-

import copy
import numpy
from PyMca import ConfigDict
from PyMca import Elements

def isValidConfiguration(configuration):
    return True, "OK"

class XASSelfAttenuationCorrection(object):
    def __init__(self, configuration=None):
        self.setConfiguration(configuration)

    def setConfiguration(self, configuration):
        if configuration is None:
            self._configuration = None
            return

        good, message = isValidConfiguration(configuration)
        if not good:
            raise RuntimeError(message)
        elif good and self._configuration in [None, {}]:
            self._configuration = copy.deepcopy(configuration)
        else:
            keyList = list(self._configuration.keys())
            for key in keyList:
                if key in configuration.keys():
                    self._configuration[key] = copy.deepcopy(configuration[key])

    def getConfiguration(self):
        return copy.deepcopy(self._configuration)

    def loadConfiguration(self, filename):
        ddict = ConfigDict.ConfigDict()
        ddict.read(filename)
        self.setConfiguration(ddict)

    def saveConfiguration(self, filename):
        ddict = ConfigDict.ConfigDict()
        config = self.getConfiguration()
        for key in config.keys():
            ddict[key] = config[key]
        ddict.write(filename)

    def correctNormalizedSpectrum(self, energy0, spectrum):
        """
        """
        element = self._configuration['XAS']['element']
        material = self._configuration['XAS'].get('material', element)
        edge = self._configuration['XAS']['edge']
        alphaIn, alphaOut = self._configuration['XAS']['angles']
        edgeEnergy = Elements.Element[element]['binding'][edge]
        userEdgeEnergy = self._configuration['XAS'].get('energy', edgeEnergy)
        energy = numpy.array(energy0, dtype=numpy.float)
    
        #PyMca data ar in keV but XAS data are usually in eV
        if 0.5 * (energy[0] + energy[-1])/edgeEnergy > 100:
            # if the user did not do stupid things most likely
            # the energy was given in eV
            energy *= 0.001

        if userEdgeEnergy/edgeEnergy > 100:
            userEdgeEnergy *= 0.001

        # forget about multilayers for the time being
        # Elements.getMaterialMassFractions(materialList, massFractionsList)
        massFractions = Elements.getMaterialMassFractions([material], [1.0])

        # calculate the total mass attenuation coefficients at the given energies
        # exciting the given element shell and not exciting it
        EPDL = Elements.PyMcaEPDL97
        totalCrossSection = 0.0
        totalCrossSectionBackground = 0.0
        for ele in massFractions.keys():
            # make sure EPDL97 respects the Elements energies
            if EPDL.EPDL97_DICT[ele]['original']:
                EPDL.setElementBindingEnergies(ele,
                                               Elements.Element[ele]['binding'])
            if ele == element:
                # make sure we respect the user energy
                if abs(userEdgeEnergy-edgeEnergy) > 0.01:
                    newBinding = Elements.Element[ele]['binding']
                    newBinding[edge] = userEdgeEnergy
                    try:
                        EPDL.setElementBindingEnergies(ele, newBinding)
                        crossSections = EPDL.getElementCrossSections(ele, energy)
                        EPDL.setElementBindingEnergies(ele,
                                                       Elements.Element[ele]['binding'])
                    except:
                        EPDL.setElementBindingEnergies(ele,
                                                       Elements.Element[ele]['binding'])
                        raise
                else:
                    crossSections = EPDL.getElementCrossSections(ele, energy)
            else:
                crossSections = EPDL.getElementCrossSections(ele, energy)
            total = numpy.array(crossSections['total'])
            tmpFloat = massFractions[ele] * total
            totalCrossSection += tmpFloat
            if ele != element:
                totalCrossSectionBackground += tmpFloat
            else:
                edgeCrossSections = numpy.array(crossSections[edge])
                muSampleJump = massFractions[ele] * edgeCrossSections
                totalCrossSectionBackground += massFractions[ele] *\
                                            (total - edgeCrossSections)
        # calculate the mass attenuation coefficient of the sample at the fluorescent energy
        # assume we are detecting the main fluorescence line of the element shell
        if edge == 'K':
            rays = Elements.Element[element]['Ka rays']
        elif edge[0] == 'L':
            rays = Elements.Element[element][edge + " xrays"]
        elif edge[0] == 'M':
            rays = []
            for transition in Elements.Element[element]['M xrays']:
                if transition.startswith(edge):
                    rays.append(transition)

        lineList = []
        for label in rays:
            ene  = Elements.Element[element][label]['energy']
            rate = Elements.Element[element][label]['rate']
            lineList.append([ene, rate, label])

        # whithin 50 eV lines considered the same
        lineList = Elements._filterPeaks(lineList, ethreshold=0.050)

        # now take the returned line with the highest intensity
        fluoLine = lineList[0]
        for line in lineList:
            if line[1] > fluoLine[1]:
                fluoLine = line

        # and calculate the sample total mass attenuation
        muTotalFluorescence = 0.0
        for ele in massFractions.keys():
            crossSections = EPDL.getElementCrossSections(ele, fluoLine[0])
            muTotalFluorescence += massFractions[ele] * crossSections['total'][0]
    
        
        #FORMULA ANA ========================================================================
        sinIn = numpy.sin(numpy.deg2rad(alphaIn))
        sinOut= numpy.sin(numpy.deg2rad(alphaOut))
        g = sinIn / sinOut
        if 1:
            idx = numpy.where(muSampleJump > 0.0)[0][0]
            muSampleJump[0:idx] = muSampleJump[idx]
            ALPHA = g * (muTotalFluorescence/muSampleJump) + totalCrossSectionBackground/muSampleJump
            return (spectrum * ALPHA)/(1 + ALPHA - spectrum)
        elif 0:
            d = thickness * density #
            idx = numpy.where(muSampleJump > 0.0)[0][0]
            muSampleJump[0:idx] = muSampleJump[idx]
            ALPHA = g * (muTotalFluorescence/muSampleJump) + totalCrossSectionBackground/muSampleJump
            thickTarget0 = (spectrum * ALPHA)/(1 + ALPHA - spectrum)
            # Iterate to find the solution
            x = spectrum
            t = (ALPHA + 1)  * d * muSampleJump/sinIn
            if t.max() < 0.001:
                print("A Approximated")
                A = 1 - t
            else:
                A = numpy.exp(-t)
            t = (ALPHA * d * muSampleJump/sinIn)
            if t.max() < 0.001:
                print("B Approximated")
                B = 1.0 - t
            else:
                B = numpy.exp(-t)
            delta = 10.0
            i = 0
            while (delta > 1.0e-5) and (i < 30):
                old = x
                x = thickTarget0 * (1.0 - A) / \
                                   (1.0 - B * numpy.exp(- x * d * muSampleJump/sinIn))
                delta = numpy.abs(x - old).max()
                i += 1
                print "%d delta %f" % (i, delta)
            return x
        elif 0:
            # FORMULA Booth and Bridges for infinite sample
            idx = numpy.where(muSampleJump > 0.0)[0][0]
            muSampleJump[0:idx] = muSampleJump[idx]
            ALPHA =  g * muTotalFluorescence + totalCrossSection
            return (spectrum - (totalCrossSection/ALPHA) * (spectrum - 1) - (muSampleJump/ALPHA))/\
                   (1.0 - (totalCrossSection/ALPHA) * (spectrum - 1) - (muSampleJump/ALPHA))
        else:
            thickness = 1.0
            density = 1.0e-6
            # FORMULA Booth and Bridges
            ALPHA =  g * muTotalFluorescence + totalCrossSection
            tmpFloat0 = density * thickness * ALPHA / sinIn
            tmpFloat1 = numpy.exp(-tmpFloat0)
            BETA = (muSampleJump * tmpFloat0) * tmpFloat1
            print "BETA = ", BETA
            GAMMA = 1.0 - tmpFloat1
            b = GAMMA * ( ALPHA  - muSampleJump * spectrum + BETA)
            discriminant = b*b + 4 * ALPHA * BETA * GAMMA * (spectrum - 1.0)
            return 1 + (-b + numpy.sqrt(discriminant))/(2 * BETA)
        
if __name__ == "__main__":
    from PyMca import specfilewrapper
    instance = XASSelfAttenuationCorrection()
    configuration = {}
    configuration['XAS'] = {}
    configuration['XAS']['material'] = 'Pd'
    configuration['XAS']['element'] = 'Pd'
    configuration['XAS']['edge'] = 'L3'
    configuration['XAS']['energy'] = Elements.Element['Pd']['binding']['L3']
    configuration['XAS']['angles'] = [45., 45.]
    instance.setConfiguration(configuration)

    normalizedFile = specfilewrapper.Specfile('norm501.dat')
    normalizedScan = normalizedFile[0]
    energy, spectrum = normalizedScan[0], normalizedScan[1]
    normalizedScan = None
    normalizedFile = None
    correctedSpectrum = instance.correctNormalizedSpectrum(energy, spectrum)

    from matplotlib import pyplot as pl
    pl.plot(energy, spectrum, 'b')
    pl.plot(energy, correctedSpectrum, 'r')
    pl.show()

    normalizedFile = specfilewrapper.Specfile('PdL3Fabrice.DAT')
    normalizedScan = normalizedFile[0]
    data = normalizedScan.data()
    energy = data[1, :]
    spectrum = data[2, :]
    corr = data[3, :]

    normalizedScan = None
    normalizedFile = None
    correctedSpectrum = instance.correctNormalizedSpectrum(energy, spectrum)

    pl.plot(energy, spectrum, 'b')
    pl.plot(energy, corr, 'y')
    pl.plot(energy, correctedSpectrum, 'r')
    pl.show()

