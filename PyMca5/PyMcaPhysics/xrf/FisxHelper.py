#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
import os
from fisx import DataDir
from fisx import Elements as FisxElements
from fisx import Material
from fisx import Detector
from fisx import XRF
dataDir = DataDir.DATA_DIR
bindingEnergies = os.path.join(dataDir, "BindingEnergies.dat")
xcomFile = os.path.join(dataDir, "XCOM_CrossSections.dat")
xcom = FisxElements(dataDir, bindingEnergies, xcomFile)

def getMultilayerFluorescence(multilayerSample,
                              energyList,
                              layerList = None,
                              weightList=None,
                              flagList  = None,
                              fulloutput = None,
                              beamFilters = None,
                              elementsList = None,
                              attenuatorList  = None,
                              alphaIn      = None,
                              alphaOut     = None,
                              cascade = None,
                              detector= None,
                              elementsFromMatrix=False,
                              secondary=None,
                              materials=None):
    if materials is not None:
        for material in materials:
            xcom.addMaterial(material, errorOnReplace=0)

    # the instance
    xrf = XRF()

    # the beam energies
    if not len(energyList):
        raise ValueError("Empty list of beam energies!!!")
    xrf.setBeam(energyList, weights=weightList,
                            characteristic=flagList)
    # the beam filters (if any)
    if beamFilters is None:
        beamFilters = []
    """
    Due to wrapping constraints, the filter list must have the form:
    [[Material name or formula0, density0, thickness0, funny factor0],
     [Material name or formula1, density1, thickness1, funny factor1],
     ...
     [Material name or formulan, densityn, thicknessn, funny factorn]]

    Unless you know what you are doing, the funny factors must be 1.0
    """
    xrf.setBeamFilters(beamFilters)

    # the sample description
    """
    Due to wrapping constraints, the list must have the form:
    [[Material name or formula0, density0, thickness0, funny factor0],
     [Material name or formula1, density1, thickness1, funny factor1],
     ...
     [Material name or formulan, densityn, thicknessn, funny factorn]]

    Unless you know what you are doing, the funny factors must be 1.0
    """
    xrf.setSample(multilayerSample)

    # the attenuators
    if attenuatorList is not None:
        if len(attenuatorList) > 0:
            xrf.setAttenuators(attenuatorList)

    # the geometry
    if alphaIn is None:
        alphaIn = 45
    if alphaOut is None:
        alphaOut = 45
    xrf.setGeometry(alphaIn, alphaOut)

    # the detector
    if detector is not None:
        # Detector must be a list as [material, density, thickness]
        if len(detector) > 0:
            if len(detector) == 3:
                detectorInstance = Detector(detector[0],
                                            density=detector[1],
                                            thickness=detector[2])
            else:
                detectorInstance = Detector(detector[0],
                                            density=detector[1],
                                            thickness=detector[2],
                                            funny=detector[3])
        xrf.setDetector(detectorInstance)
    else:
        print("DETECTOR HAS TO BE REMOVED!!!!")

    if elementsFromMatrix:
        newElementsList = []
        for peak in elementsList:
            ele = peak.split()[0]
            considerIt = False
            for layer in multilayerSample:
                composition = xcom.getComposition(layer[0])
                if ele in composition:
                    considerIt = True
            if considerIt:
                newElementsList.append(peak)
        elementsList = newElementsList

    # the detector distance and solid angle ???
    if elementsList in [None, []]:
        raise ValueError("Element list not specified")

    if len(elementsList):
        if len(elementsList[0]) == 3:
            actualElementList = [x[1] + " " + x[2] for x in elementsList]
        elif len(elementsList[0]) == 2:
            actualElementList = [x[0] + " " + x[1] for x in elementsList]
        else:
            actualElementList = elementsList
    else:
        actualElementList = []
        i = 0
        for layer in multilayerSample:
            composition = xcom.getComposition(layer[0])
            for element in composition.keys():
                actualElementList.append("%s %s %d" % (element, "K", i))
            i += 1

    return xrf.getMultilayerFluorescence(actualElementList, xcom, \
                                         secondary=secondary,
                                         useMassFractions=elementsFromMatrix)

def _getFisxMaterials(fitConfiguration):
    """
    Given a PyMca fir configuration, return the list of fisx materials to be
    used by the library for calculation purposes.
    """
    # define all the needed materials
    inputMaterialDict = fitConfiguration.get("materials", {})
    inputMaterialList = list(inputMaterialDict.keys())
    nMaterials = len(inputMaterialList)
    fisxMaterials = []
    processedMaterialList = []

    while (len(processedMaterialList) != nMaterials):
        for i in range(nMaterials):
            materialName = inputMaterialList[i]
            if materialName in processedMaterialList:
                # already defined
                pass
            else:
                thickness = inputMaterialDict[materialName].get("Thickness", 1.0)
                density = inputMaterialDict[materialName].get("Density", 1.0)
                comment = inputMaterialDict[materialName].get("Comment", "")
                compoundList = inputMaterialDict[materialName]["CompoundList"]
                fractionList = inputMaterialDict[materialName]["CompoundFraction"]
                if not hasattr(fractionList, "__getitem__"):
                    compoundList = [compoundList]
                    fractionList = [fractionList]
                composition = {}
                for n in range(len(compoundList)):
                    composition[compoundList[n]] = fractionList[n]
                # check the composition is expressed in terms of elements
                # and not in terms of other undefined materials
                totallyDefined = True
                for element in composition:
                    #check if it can be understood
                    if not len(xcom.getComposition(element)):
                        # compound not understood
                        # probably we have a material defined in terms of other material
                        totallyDefined = False
                if totallyDefined:
                  fisxMaterial = Material(materialName,
                                          density=density,
                                          thickness=thickness,
                                          comment=comment)
                  fisxMaterial.setComposition(composition)
                  fisxMaterials.append(fisxMaterial)
                  processedMaterialList.append(materialName)
    return fisxMaterials

def _getBeam(fitConfiguration):
    inputEnergy = fitConfiguration['fit'].get('energy', None)
    if inputEnergy in [None, ""]:
        raise ValueError("Beam energy has to be specified")
    if not hasattr(inputEnergy, "__len__"):
        energyList = [inputEnergy]
        weightList = [1.0]
        characteristicList = [1]
    else:
        energyList    = []
        weightList  = []
        characteristicList = []
        for i in range(len(inputEnergy)):
            if fitConfiguration['fit']['energyflag'][i]:
                energy = fitConfiguration['fit']['energy'][i]
                if energy is not None:
                    if fitConfiguration['fit']['energyflag'][i]:
                        weight = fitConfiguration['fit']['energyweight'][i]
                        if weight > 0.0:
                            energyList.append(energy)
                            weightList.append(weight)
                    if 'energyscatter' in fitConfiguration['fit']:
                        characteristicList.append(fitConfiguration['fit'] \
                                              ['energyscatter'][i])
                    elif i == 0:
                        characteristicList.append(1)
                    else:
                        characteristicList.append(0)
    return energyList, weightList, characteristicList

def _getFiltersMatrixAttenuatorsDetectorGeometry(fitConfiguration):
    useMatrix = False
    attenuatorList =[]
    filterList = []
    detector = None
    for attenuator in list(fitConfiguration['attenuators'].keys()):
        if not fitConfiguration['attenuators'][attenuator][0]:
            # set to be ignored
            continue
        #if len(fitConfiguration['attenuators'][attenuator]) == 4:
        #    fitConfiguration['attenuators'][attenuator].append(1.0)
        if attenuator.upper() == "MATRIX":
            if fitConfiguration['attenuators'][attenuator][0]:
                useMatrix = True
                matrix = fitConfiguration['attenuators'][attenuator][1:4]
                alphaIn= fitConfiguration['attenuators'][attenuator][4]
                alphaOut= fitConfiguration['attenuators'][attenuator][5]
            else:
                useMatrix = False
        elif attenuator.upper() == "DETECTOR":
            detector = fitConfiguration['attenuators'][attenuator][1:5]
        elif attenuator.upper().startswith("BEAMFILTER"):
            filterList.append(fitConfiguration['attenuators'][attenuator][1:5])
        else:
            attenuatorList.append(fitConfiguration['attenuators'][attenuator][1:5])
    if not useMatrix:
        raise ValueError("Sample matrix has to be specified!")

    if matrix[0].upper() == "MULTILAYER":
        multilayerSample = []
        layerKeys = list(fitConfiguration['multilayer'].keys())
        if len(layerKeys):
            layerKeys.sort()
        for layer in layerKeys:
            if fitConfiguration['multilayer'][layer][0]:
                multilayerSample.append(fitConfiguration['multilayer'][layer][1:])
    else:
        multilayerSample = [matrix]
    return filterList, multilayerSample, attenuatorList, detector, \
           alphaIn, alphaOut

def _getPeakList(fitConfiguration):
    elementsList = []
    for element in fitConfiguration['peaks'].keys():
        if len(element) > 1:
            ele = element[0:1].upper() + element[1:2].lower()
        else:
            ele = element.upper()
        if type(fitConfiguration['peaks'][element]) == type([]):
            for peak in fitConfiguration['peaks'][element]:
                elementsList.append(ele + " " + peak)
        else:
            for peak in [fitConfiguration['peaks'][element]]:
                elementsList.append(ele + " " + peak)
    elementsList.sort()
    return elementsList

def getMultilayerFluorescenceFromFitConfiguration(fitConfiguration,
                                                  elementsFromMatrix=False):
    return _fisxFromFitConfigurationAction(fitConfiguration,
                                        action="fluorescence",
                                        elementsFromMatrix=elementsFromMatrix)

def getFisxCorrectionFactorsFromFitConfiguration(fitConfiguration,
                                                  elementsFromMatrix=False):
    return _fisxFromFitConfigurationAction(fitConfiguration,
                                        action="correction",
                                        elementsFromMatrix=elementsFromMatrix)

def _fisxFromFitConfigurationAction(fitConfiguration,
                                    action=None,
                                    elementsFromMatrix=False):
    if action is None:
        raise ValueError("Please specify action")
    # This is highly inefficient because one has to perform all the parsing
    # that has been already made when configuring the fit. However, this is
    # currently the simplest implementation that can work as standalone given
    # the fit configuration

    # the fisx materials list
    fisxMaterials = _getFisxMaterials(fitConfiguration)

    # extract beam parameters
    energyList, weightList, characteristicList = _getBeam(fitConfiguration)

    # extract beamFilters, matrix, geometry, attenuators and detector
    filterList, multilayerSample, attenuatorList, detector, alphaIn, alphaOut \
                = _getFiltersMatrixAttenuatorsDetectorGeometry(fitConfiguration)

    # The elements and families to be considered
    elementsList = _getPeakList(fitConfiguration)

    if action.upper() == "FLUORESCENCE":
        return getMultilayerFluorescence(multilayerSample,
                                      energyList,
                                      weightList = weightList,
                                      flagList = characteristicList,
                                      fulloutput = None,
                                      beamFilters = filterList,
                                      elementsList = elementsList,
                                      attenuatorList = attenuatorList,
                                      alphaIn = alphaIn,
                                      alphaOut = alphaOut,
                                      cascade = None,
                                      detector = detector,
                                      elementsFromMatrix=elementsFromMatrix,
                                      secondary=True,
                                      materials=fisxMaterials)
    else:
        return getFisxCorrectionFactors(multilayerSample,
                                      energyList,
                                      weightList = weightList,
                                      flagList = characteristicList,
                                      fulloutput = None,
                                      beamFilters = filterList,
                                      elementsList = elementsList,
                                      attenuatorList = attenuatorList,
                                      alphaIn = alphaIn,
                                      alphaOut = alphaOut,
                                      cascade = None,
                                      detector = detector,
                                      elementsFromMatrix=elementsFromMatrix,
                                      secondary=True,
                                      materials=fisxMaterials)

def getFisxCorrectionFactors(*var, **kw):
    expectedFluorescence = getMultilayerFluorescence(*var, **kw)
    ddict = {}
    transitions = ['K', 'Ka', 'Kb', 'L', 'L1', 'L2', 'L3', 'M']
    for key in expectedFluorescence:
        element, family = key.split()
        if element not in ddict:
            ddict[element] = {}
        if family not in transitions:
            raise KeyError("Invalid transition family: " + family)
        if family not in ddict[element]:
            ddict[element][family] = {'total':0.0,
                                      'correction_factor':[1.0, 1.0],
                                      'counts':[0.0, 0.0]}
        for iLayer in range(len(expectedFluorescence[key])):
            layerOutput = expectedFluorescence[key][iLayer]
            for line in layerOutput:
                rate = layerOutput[line]["rate"]
                primary = layerOutput[line]["primary"]
                secondary = layerOutput[line]["secondary"]
                if rate <= 0.0:
                    continue
                # primary counts
                ddict[element][family]["counts"][0] += \
                                            rate * (primary / (primary + secondary))
                ddict[element][family]["counts"][1] += rate
                ddict[element][family]["total"] += rate
    for element in ddict:
        for family in ddict[element]:
            # only second order for the time being
            firstOrder = ddict[element][family]["counts"][0]
            secondOrder = ddict[element][family]["counts"][1]
            ddict[element][family]["correction_factor"][1] = \
                       secondOrder / firstOrder
    return ddict

def getFisxCorrectionFactorsFromFitConfigurationFile(fileName):
    from PyMca5.PyMca import ConfigDict
    d = ConfigDict.ConfigDict()
    d.read(fileName)
    return getFisxCorrectionFactorsFromFitConfiguration(d)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python FisxHelper FitConfigurationFile")
    fileName = sys.argv[1]
    print(getFisxCorrectionFactorsFromFitConfigurationFile(fileName))

