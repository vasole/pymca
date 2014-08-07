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
                              forcepresent=None,
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
            if len(attenuatorList[0]) != 4:
                raise TypeError("Attenuators must include funny factor term")
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
        print("elementsList = ", elementsList)
        ele = "Fe K 0"
        actualElementList = ["Fe K 0"]
    return xrf.getMultilayerFluorescence(actualElementList, xcom, \
                                         secondary=secondary)


def getFisxCorrectionFactorsFromFitConfiguration(fitConfiguration):
    # This is highly inefficient because one has to perform all the parsing
    # that has been already made when configuring the fit. However, this is
    # currently the simplest implementation that can work as standalone given
    # the fit configuration

    # the list of defined elements
    elementsList = xcom.getElementNames()

    # define all the needed materials
    inputMaterialDict = fitConfiguration.get("materials", {})
    inputMaterialList = list(inputMaterialDict.keys())
    nMaterials = len(inputMaterialList)
    fisxMaterials = []
    processedMaterialList = []

    while (len(processedMaterialList) != nMaterials):
        print processedMaterialList
        for i in range(nMaterials):
            materialName = inputMaterialList[i]
            if materialName in processedMaterialList:
                # already defined
                pass
            else:
                print("processing ", materialName)
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

    # extract beam parameters
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
    #    xrf.setBeam(energyList, weights=weightList,
    #                        characteristic=characteristicList)

    # extract beamFiltes, matrix, geometry, attenuators and detector
    useMatrix = False
    attenuatorList =[]
    filterList = []
    detector = None
    for attenuator in list(fitConfiguration['attenuators'].keys()):
        if not fitConfiguration['attenuators'][attenuator][0]:
            # set to be ignored
            continue
        if len(fitConfiguration['attenuators'][attenuator]) == 4:
            fitConfiguration['attenuators'][attenuator].append(1.0)
        if attenuator.upper() == "MATRIX":
            if fitConfiguration['attenuators'][attenuator][0]:
                useMatrix = True
                matrix = fitConfiguration['attenuators'][attenuator][1:4]
                alphaIn= fitConfiguration['attenuators'][attenuator][4]
                alphaOut= fitConfiguration['attenuators'][attenuator][5]
            else:
                useMatrix = False
        elif attenuator.upper() == "DETECTOR":
            detector = fitConfiguration['attenuators'][attenuator][1:]
        elif attenuator.upper()[0:-1] == "BEAMFILTER":
            filterList.append(fitConfiguration['attenuators'][attenuator][1:])
        else:
            attenuatorList.append(fitConfiguration['attenuators'][attenuator][1:])
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

    # The elements and families to be considered
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
                              forcepresent=None,
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
            ddict[element][family]["correction_factor"] = \
                       secondOrder / firstOrder
    return ddict

def getFisxCorrectionFactorsFromFitConfigurationFile(fileName):
    from PyMca5.PyMca import ConfigDict
    d = ConfigDict.ConfigDict()
    d.read(fileName)
    return getFisxCorrectionFactorsFromFitConfiguration(d)

"""
import numpy
def testThickSample():
    #both angles are 45
    omega = 0.3546
    muPhoto = 41.85
    muTotal0 = 46.7434
    muTotalF = 145.63
    # air
    TAir = numpy.exp(-19.2544*0.0012648*5)
    # Be
    TBe = numpy.exp(-2.088*1.848*0.002)
    # efficiency
    Ed = 1.0 - numpy.exp(-122.06*2.33*0.0350)
    return ((omega * muPhoto) / (muTotal0+muTotalF)) * TAir * TBe * Ed
"""

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python FisxHelper FitConfigurationFile")
    fileName = sys.argv[1]
    print(getFisxCorrectionFactorsFromFitConfigurationFile(fileName))

