#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
import sys
import time
from fisx import DataDir
from fisx import Elements as FisxElements
from fisx import Material
from fisx import Detector
from fisx import XRF
xcom = None
DEBUG = 0

def getElementsInstance(dataDir=None, bindingEnergies=None, xcomFile=None):
    if dataDir is None:
        dataDir = DataDir.FISX_DATA_DIR
    try:
        from PyMca5.PyMcaDataDir import PYMCA_DATA_DIR as pymcaDataDir
        from PyMca5 import getDataFile
    except:
        print("Using fisx shell constants and ratios")
        pymcaDataDir = None
    if bindingEnergies is None:
        if pymcaDataDir is None:
            bindingEnergies = os.path.join(dataDir, "BindingEnergies.dat")
        else:
            bindingEnergies = getDataFile("BindingEnergies.dat")
    if xcomFile is None:
        if pymcaDataDir is None:
            xcomFile = os.path.join(dataDir, "XCOM_CrossSections.dat")
        else:
            xcomFile = getDataFile("XCOM_CrossSections.dat")
    if DEBUG:
        t0 = time.time()
    instance = FisxElements(dataDir, bindingEnergies, xcomFile)
    if DEBUG:
        print("Shell constants")
    # the files should be taken from PyMca to make sure the same data are used
    for key in ["K", "L", "M"]:
        fname = instance.getShellConstantsFile(key)
        if sys.version > '3.0':
            # we have to make sure we have got a string
            if hasattr(fname, "decode"):
                fname = fname.decode("latin-1")
        if DEBUG:
            print("Before %s" % fname)
        if pymcaDataDir is not None:
            fname = getDataFile(key + "ShellConstants.dat")
        else:
            fname = os.path.join(os.path.dirname(fname),
                                 key + "ShellConstants.dat")
        instance.setShellConstantsFile(key, fname)
        if DEBUG:
            print("After %s" % instance.getShellConstantsFile(key))
    if DEBUG:
        print("Radiative transitions")

    for key in ["K", "L", "M"]:
        fname = instance.getShellRadiativeTransitionsFile(key)
        if sys.version > '3.0':
            # we have to make sure we have got a string ...
            if hasattr(fname, "decode"):
                fname = fname.decode("latin-1")
        if DEBUG:
            print("Before %s" % fname) 
        if pymcaDataDir is not None:
            fname = getDataFile(key + "ShellRates.dat")
        else:
            fname = os.path.join(os.path.dirname(fname), key + "ShellRates.dat")
        instance.setShellRadiativeTransitionsFile(key, fname)
        if DEBUG:
            print("After %s " % instance.getShellRadiativeTransitionsFile(key))

    if DEBUG:
        print("Reading Elapsed = ", time.time() - t0)
    return instance

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
                              materials=None,
                              secondaryCalculationLimit=0.0):
    if DEBUG:
        print("Library actually using secondary = ", secondary)
    global xcom
    if xcom is None:
        if DEBUG:
            print("Getting fisx elements instance")
        xcom = getElementsInstance()

    if materials is not None:
        if DEBUG:
            print("Deleting materials")
        xcom.removeMaterials()
        for material in materials:
            if DEBUG:
                print("Adding material making sure no duplicates")
            xcom.addMaterial(material, errorOnReplace=1)

    # the instance
    if DEBUG:
        print("creating XRF instance")
    xrf = XRF()

    # the beam energies
    if not len(energyList):
        raise ValueError("Empty list of beam energies!!!")
    if DEBUG:
        print("setting beam")
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
    if DEBUG:
        print("setting beamFilters")
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
    if DEBUG:
        print("setting sample")
    xrf.setSample(multilayerSample)

    # the attenuators
    if attenuatorList is not None:
        if len(attenuatorList) > 0:
            if DEBUG:
                print("setting attenuators")
            xrf.setAttenuators(attenuatorList)

    # the geometry
    if DEBUG:
        print("setting Geometry")
    if alphaIn is None:
        alphaIn = 45
    if alphaOut is None:
        alphaOut = 45
    xrf.setGeometry(alphaIn, alphaOut)

    # the detector
    if DEBUG:
        print("setting Detector")
    if detector is not None:
        # Detector can be a list as [material, density, thickness]
        # or a Detector instance
        if isinstance(detector, Detector):
            detectorInstance = detector
        elif len(detector) == 3:
            detectorInstance = Detector(detector[0],
                                            density=detector[1],
                                            thickness=detector[2])
        else:
            detectorInstance = Detector(detector[0],
                                            density=detector[1],
                                            thickness=detector[2],
                                            funny=detector[3])
    else:
        detectorInstance = Detector("")

    xrf.setDetector(detectorInstance)
    if detectorInstance.getActiveArea() > 0.0:
        useGeometricEfficiency = 1
    else:
        useGeometricEfficiency = 0

    matrixElementsList = []
    for peak in elementsList:
        ele = peak.split()[0]
        considerIt = False
        for layer in multilayerSample:
            composition = xcom.getComposition(layer[0])
            if ele in composition:
                considerIt = True
        if considerIt:
            if peak not in matrixElementsList:
                matrixElementsList.append(peak)

    if elementsFromMatrix:
        elementsList = matrixElementsList

    # the detector distance and solid angle ???
    if elementsList in [None, []]:
        raise ValueError("Element list not specified")

    if len(elementsList):
        if len(elementsList[0]) == 3:
            # PyMca can send [atomic number, element, peak]
            actualElementList = [x[1] + " " + x[2] for x in elementsList]
        elif len(elementsList[0]) == 2:
            actualElementList = [x[0] + " " + x[1] for x in elementsList]
        else:
            actualElementList = elementsList

    # enabling the cache gets a (miserable) 15 % speed up
    if DEBUG:
        print("Using cascade cache")
    for layer in multilayerSample:
        composition = xcom.getComposition(layer[0])
        for element in composition.keys():
            xcom.setElementCascadeCacheEnabled(element, 1)
    for element in actualElementList:
        xcom.setElementCascadeCacheEnabled(element.split()[0], 1)

    if DEBUG:
        print("Calling getMultilayerFluorescence")
        t0 = time.time()
    expectedFluorescence = xrf.getMultilayerFluorescence(actualElementList,
                            xcom,
                            secondary=secondary,
                            useGeometricEfficiency=useGeometricEfficiency,
                            useMassFractions=elementsFromMatrix,
                            secondaryCalculationLimit=secondaryCalculationLimit)
    if DEBUG:
        print("C++ elapsed TWO = ", time.time() - t0)

    if not elementsFromMatrix:
        # If one element was present in one layer and not on others, PyMca only
        # calculated contributions from the layers in which the element was
        # present
        for peakFamily in matrixElementsList:
            element, family = peakFamily.split()[0:2]
            key = element + " " + family
            if key in expectedFluorescence:
                # those layers where the amount of the material was 0 have
                # to present no contribution
                for iLayer in range(len(expectedFluorescence[key])):
                    layerMaterial = multilayerSample[iLayer][0]
                    if element not in xcom.getComposition(layerMaterial):
                        expectedFluorescence[key][iLayer] = {}
    return expectedFluorescence

def _getFisxMaterials(fitConfiguration):
    """
    Given a PyMca fir configuration, return the list of fisx materials to be
    used by the library for calculation purposes.
    """
    global xcom
    if xcom is None:
        xcom = getElementsInstance()

    # define all the needed materials
    inputMaterialDict = fitConfiguration.get("materials", {})
    inputMaterialList = list(inputMaterialDict.keys())
    nMaterials = len(inputMaterialList)
    fisxMaterials = []
    processedMaterialList = []
    nIter = 10000
    while (len(processedMaterialList) != nMaterials) and (nIter > 0):
        nIter -= 1
        for i in range(nMaterials):
            materialName = inputMaterialList[i]
            if materialName in processedMaterialList:
                # already defined
                pass
            else:
                thickness = inputMaterialDict[materialName].get("Thickness", 1.0)
                density = inputMaterialDict[materialName].get("Density", 1.0)
                comment = inputMaterialDict[materialName].get("Comment", "")
                if type(comment) in [type([])]:
                    # the user may have put a comma in the comment leading to
                    # misinterpretation of the string as a list
                    if len(comment) == 0:
                        comment = ""
                    elif len(comment) == 1:
                        comment = comment[0]
                    else:
                        actualComment = comment[0]
                        for commentPiece in comment[1:]:
                            actualComment = actualComment + "," + commentPiece
                        comment = actualComment
                if not len(comment):
                    comment = ""
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
                    if element in processedMaterialList:
                        # already defined
                        continue
                    elif not len(xcom.getComposition(element)):
                        # compound not understood
                        # probably we have a material defined in terms of other material
                        totallyDefined = False
                if totallyDefined:
                    try:
                        fisxMaterial = Material(materialName,
                                              density=density,
                                              thickness=thickness,
                                              comment=comment)
                        fisxMaterial.setComposition(composition)
                        fisxMaterials.append(fisxMaterial)
                    except:
                        if len(materialName):
                            raise TypeError("Error defining material <%s>" % \
                                            materialName)
                    processedMaterialList.append(materialName)
    if len(processedMaterialList) < nMaterials:
        txt = "Undefined materials. "
        for material in inputMaterialList:
            if material not in processedMaterialList:
                txt += "\nCannot define material <%s>\nComposition " % material
                compoundList = inputMaterialDict[material]["CompoundList"]
                fractionList = inputMaterialDict[material]["CompoundFraction"]
                for compound in compoundList:
                    if not len(xcom.getComposition(compound)):
                        if compound not in processedMaterialList:
                            if compound + " " in processedMaterialList:
                                txt += "contains <%s> (defined as <%s>), " % (compound, compound + " ")
                            else:
                                txt += "contains <%s> (undefined)," % compound
        print(txt)
        raise KeyError(txt)
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

def _getFisxDetector(fitConfiguration, attenuatorsDetector=None):
    distance = fitConfiguration["concentrations"]["distance"]
    area = fitConfiguration["concentrations"]["area"]
    detectorMaterial = fitConfiguration["detector"]["detele"]

    if attenuatorsDetector is None:
        # user is not interested on accounting for detection efficiency
        if fitConfiguration["fit"]["escapeflag"]:
            # but wants to account for escape peaks
            # we can forget about efficiency but not about detector composition
            # assign "infinite" efficiency
            density = 0.0
            thickness = 0.0
            fisxDetector = Detector(detectorMaterial,
                                    density=density,
                                    thickness=thickness)
        else:
            # user is not interested on considering the escape peaks
            fisxDetector = None
    else:
        # make sure information is consistent
        if attenuatorsDetector[0] not in [detectorMaterial, detectorMaterial+"1"]:
            print("%s not equal to %s" % (attenuatorsDetector[0], detectorMaterial))
            msg = "Inconsistent detector material between DETECTOR and ATTENUATORS tab"
            msg += "\n%s not equal to %s" % (attenuatorsDetector[0], detectorMaterial)
            raise ValueError(msg)
        if len(attenuatorsDetector) == 3:
            fisxDetector = Detector(detectorMaterial,
                                density=attenuatorsDetector[1],
                                thickness=attenuatorsDetector[2])
        else:
            fisxDetector = Detector(detectorMaterial,
                                density=attenuatorsDetector[1],
                                thickness=attenuatorsDetector[2],
                                funny=attenuatorsDetector[3])
        fisxDetector.setActiveArea(area)
        fisxDetector.setDistance(distance)
    if fisxDetector is not None:
        nThreshold = fitConfiguration["detector"]["nthreshold"]
        fisxDetector.setMaximumNumberOfEscapePeaks(nThreshold)
    return fisxDetector

def getMultilayerFluorescenceFromFitConfiguration(fitConfiguration,
                                                  elementsFromMatrix=False,
                                                  secondaryCalculationLimit=0.0):
    return _fisxFromFitConfigurationAction(fitConfiguration,
                                        action="fluorescence",
                                        elementsFromMatrix=elementsFromMatrix,
                                        secondaryCalculationLimit= \
                                           secondaryCalculationLimit)

def getFisxCorrectionFactorsFromFitConfiguration(fitConfiguration,
                                                 elementsFromMatrix=False,
                                                 secondaryCalculationLimit=0.0):
    return _fisxFromFitConfigurationAction(fitConfiguration,
                                        action="correction",
                                        elementsFromMatrix=elementsFromMatrix,
                                        secondaryCalculationLimit= \
                                           secondaryCalculationLimit)

def _fisxFromFitConfigurationAction(fitConfiguration,
                                    action=None,
                                    elementsFromMatrix=False, \
                                    secondaryCalculationLimit=0.0):
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

    # The detection setup
    detectorInstance = _getFisxDetector(fitConfiguration, detector)

    try:
        secondary = fitConfiguration["concentrations"]["usemultilayersecondary"]
        if secondary == 0:
            # otherways it is meaning less to call this function
            secondary = 2
    except:
        print("Exception. Forcing tertiary")
        secondary = 2

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
                                      detector = detectorInstance,
                                      elementsFromMatrix=elementsFromMatrix,
                                      secondary=secondary,
                                      materials=fisxMaterials,
                                      secondaryCalculationLimit= \
                                          secondaryCalculationLimit)
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
                                      detector = detectorInstance,
                                      elementsFromMatrix=elementsFromMatrix,
                                      secondary=secondary,
                                      materials=fisxMaterials,
                                      secondaryCalculationLimit= \
                                          secondaryCalculationLimit)

def getFisxCorrectionFactors(*var, **kw):
    expectedFluorescence = getMultilayerFluorescence(*var, **kw)
    ddict = {}
    transitions = ['K', 'Ka', 'Kb', 'L', 'L1', 'L2', 'L3', 'M']
    if kw["secondary"] == 2:
        nItems = 3
    else:
        nItems = 2
    for key in expectedFluorescence:
        element, family = key.split()
        if element not in ddict:
            ddict[element] = {}
        if family not in transitions:
            raise KeyError("Invalid transition family: %s" % family)
        if family not in ddict[element]:
            ddict[element][family] = {'total':0.0,
                                      'correction_factor':[1.0] * nItems,
                                      'counts':[0.0] * nItems}
        for iLayer in range(len(expectedFluorescence[key])):
            layerOutput = expectedFluorescence[key][iLayer]
            layerKey = "layer %d" % iLayer
            if layerKey not in ddict[element][family]:
                ddict[element][family][layerKey] = {'total':0.0,
                                    'correction_factor':[1.0] * nItems,
                                    'counts':[0.0] * nItems}
            for line in layerOutput:
                rate = layerOutput[line]["rate"]
                primary = layerOutput[line]["primary"]
                secondary = layerOutput[line]["secondary"]
                tertiary = layerOutput[line].get("tertiary", 0.0)
                if rate <= 0.0:
                    continue
                # primary counts
                tmpDouble = rate * (primary / (primary + secondary + tertiary))
                ddict[element][family]["counts"][0] += tmpDouble
                secondaryCounts = rate * \
                    ((primary  + secondary)/ (primary + secondary + tertiary))
                ddict[element][family]["counts"][1] += secondaryCounts
                if nItems == 3:
                    ddict[element][family]["counts"][2] += rate
                ddict[element][family]["total"] += rate

                #layer by layer information
                ddict[element][family][layerKey]["counts"][0] += tmpDouble
                ddict[element][family][layerKey]["counts"][1] += secondaryCounts
                if nItems == 3:
                    ddict[element][family][layerKey]["counts"][2] += rate
                ddict[element][family][layerKey]["total"] += rate

    for element in ddict:
        for family in ddict[element]:
            # second order includes tertiary!!!
            firstOrder = ddict[element][family]["counts"][0]
            secondOrder = ddict[element][family]["counts"][1]
            ddict[element][family]["correction_factor"][1] = \
                       secondOrder / firstOrder
            if nItems == 3:
                thirdOrder = ddict[element][family]["counts"][2]
                ddict[element][family]["correction_factor"][2] = \
                                   thirdOrder / firstOrder
            i = 0
            layerKey = "layer %d" % i
            while layerKey in ddict[element][family]:
                firstOrder = ddict[element][family][layerKey]["counts"][0]
                secondOrder = ddict[element][family][layerKey]["counts"][1]
                if firstOrder <= 0:
                    if secondOrder > 0.0:
                        print("Inconsistency? secondary with no primary?")
                    ddict[element][family][layerKey]["correction_factor"][1] = 1
                    if nItems == 3:
                        ddict[element][family][layerKey]["correction_factor"][2] = 1
                else:
                    ddict[element][family][layerKey]["correction_factor"][1] =\
                       secondOrder / firstOrder
                    if nItems == 3:
                        ddict[element][family][layerKey]["correction_factor"][2] =\
                           thirdOrder / firstOrder
                i += 1
                layerKey = "layer %d" % i
    return ddict

def getFisxCorrectionFactorsFromFitConfigurationFile(fileName,
                                                     elementsFromMatrix=False,
                                                secondaryCalculationLimit=0.0):
    from PyMca5.PyMca import ConfigDict
    d = ConfigDict.ConfigDict()
    d.read(fileName)
    return getFisxCorrectionFactorsFromFitConfiguration(d,
                                        elementsFromMatrix=elementsFromMatrix,
                                        secondaryCalculationLimit= \
                                            secondaryCalculationLimit)

if __name__ == "__main__":
    DEBUG = 1
    import time
    import sys
    if len(sys.argv) < 2:
        print("Usage: python FisxHelper FitConfigurationFile [element] [matrix_flag]")
        sys.exit(0)
    fileName = sys.argv[1]
    if len(sys.argv) > 2:
        element = sys.argv[2]
        if len(sys.argv) > 3:
            matrix = int(sys.argv[3])    
            print(getFisxCorrectionFactorsFromFitConfigurationFile(\
                fileName, elementsFromMatrix=matrix))[element]
        else:
            print(getFisxCorrectionFactorsFromFitConfigurationFile(fileName)) \
                                                                    [element]
    else:
        print(getFisxCorrectionFactorsFromFitConfigurationFile(fileName))

