#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
import copy
import logging
from . import Elements
from . import ConcentrationsTool

_logger = logging.getLogger(__name__)


class SingleLayerStrategy(object):
    def __init__(self):
        self._tool = ConcentrationsTool.ConcentrationsTool()

    def applyStrategy(self, fitResult, fluorescenceRates, currentIteration=None):
        """
        Provided a fit result, it returns an new fit configuration and
        a positive integer to indicate the strategy procedure has not finished.

        Returning an empty fit configuration, or a number of iterations equal 0
        will indicate the process is over.
        """
        _logger.debug("SingleLayerStrategy called with iteration %s", currentIteration)
        newConfiguration = copy.deepcopy(fitResult['config'])
        strategyConfiguration = newConfiguration['SingleLayerStrategy']
        if currentIteration is None:
            currentIteration = strategyConfiguration['iterations']
        if currentIteration < 1:
            # enough for the journey
            return {}, 0

        # calculate concentrations with current configuration
        ddict = {}
        ddict.update(newConfiguration['concentrations'])
        ddict, addInfo = self._tool.processFitResult( \
                                config=ddict,
                                fitresult={"result":fitResult},
                                elementsfrommatrix=False,
                                fluorates = fluorescenceRates,
                                addinfo=True)

        # find the layer to be updated the matrix
        matrixKey = None
        for attenuator in list(newConfiguration['attenuators'].keys()):
            if not newConfiguration['attenuators'][attenuator][0]:
                continue
            if attenuator.upper() == "MATRIX":
                if newConfiguration['attenuators'][attenuator][1].upper() != \
                                    "MULTILAYER":
                    matrixKey = attenuator
                else:
                    matrixKey = "MULTILAYER"
            if matrixKey:
                break
        if matrixKey != "MULTILAYER":
            parentKey = 'attenuators'
            daughterKey = matrixKey
        else:
            parentKey = "multilayer"
            daughterKey = None
            if newConfiguration["SingleLayerStrategy"]["layer"].upper() == \
                                                       ["AUTO"]:
                # we have to find the layer where we should work
                firstLayer = None
                for layer in newConfiguration[parentKey]:
                    if newConfiguration[parentKey][layer][0]:
                        material = newConfiguration[parentKey][layer][1]
                        composition = Elements.getMaterialMassFractions( \
                                    [material], [1.0])
                        for group in strategyConfiguration["peaks"]:
                            if "-" in group:
                                continue
                            ele = group.split()[0]
                            if ele in composition:
                                daughterLayer = layer
                                break
                        if firstLayer is None:
                            firstLayer = layer
                if daughterKey is None:
                    daughterKey = firstLayer
            else:
                daughterKey = newConfiguration["SingleLayerStrategy"]["layer"]
            if daughterKey is None:
                raise ValueError("Cannot find appropriate sample layer")
        # newConfiguration[parentKey][daughterKey] composition is to be updated
        # get the new composition
        total = 0.0
        CompoundList = []
        CompoundFraction = []
        materialCounter = -1
        for group in strategyConfiguration["peaks"]:
            materialCounter += 1
            if "-" in group:
                continue
            if strategyConfiguration["flags"][materialCounter] in ["0", 0]:
                _logger.debug("ignoring %s", group)
                continue
            ele = group.split()[0]
            material = strategyConfiguration["materials"][materialCounter]
            if material in ["-", ele, ele + "1"]:
                CompoundList.append(ele)
                CompoundFraction.append(\
                                ddict["mass fraction"][group])
            else:
                massFractions = Elements.getMaterialMassFractions( \
                            [material], [1.0])
                CompoundFraction.append( \
                        ddict["mass fraction"][group] / massFractions[ele])
                CompoundList.append(material)
            total += CompoundFraction[-1]
        if strategyConfiguration["completer"] not in ["-"]:
            if total < 1.0:
                CompoundList.append(strategyConfiguration["completer"])
                CompoundFraction.append(1.0 - total)
            else:
                for i in range(len(CompoundFraction)):
                    CompoundFraction[i] /= total;
        else:
            for i in range(len(CompoundFraction)):
                CompoundFraction[i] /= total;
        materialName = "SingleLayerStrategyMaterial"
        newConfiguration["materials"][materialName] = \
            {"Density": newConfiguration[parentKey][daughterKey][2],
             "Thickness":newConfiguration[parentKey][daughterKey][3],
             "CompoundList":CompoundList,
             "CompoundFraction":CompoundFraction,
             "Comment":"Last Single Layer Strategy iteration"}
        # and update it
        newConfiguration[parentKey][daughterKey][1] = materialName
        _logger.debug("Updated sample material: %s",
                      newConfiguration["materials"][materialName])
        return newConfiguration, currentIteration - 1
