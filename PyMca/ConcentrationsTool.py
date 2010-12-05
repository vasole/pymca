#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
# is a problem for you.
#############################################################################*/
__revision__ = "$Revision: 1.26 $"
__author__="V.A. Sole - ESRF BLISS Group"
import Elements
import copy
import types
import numpy.oldnumeric as Numeric
import string

class ConcentrationsConversion:
    def getConcentrationsAsHtml(self, concentrations = None):
        text = ""
        if concentrations is None:return text

        result = concentrations
        #the header
        if result.has_key('mmolar'):
            mmolarflaglist = [False, True]
        else:
            mmolarflaglist = [False]

        for mmolarflag in mmolarflaglist:
            text+="\n"
            text+= "<H2><a NAME=""%s""></a><FONT color=#009999>" % 'Concentrations'
            if mmolarflag:
                text+= "%s:" % 'mM Concentrations'
            else:
                text+= "%s:" % 'Concentrations'
            text+= "</FONT></H2>"
            text+="<br>"
            if mmolarflag:
                labels = ['Element','Group','Fit Area','Sigma Area', 'mM concentration']
            else:
                labels = ['Element','Group','Fit Area','Sigma Area', 'Mass fraction']
            
            #the table
            if result.has_key('layerlist'):
                if type(result['layerlist']) != type([]):
                    result['layerlist'] = [result['layerlist']]
                for label in result['layerlist']:
                    labels += [label]
            lemmon=string.upper("#%x%x%x" % (255,250,205))
            white ='#FFFFFF' 
            hcolor = string.upper("#%x%x%x" % (230,240,249))       
            text+="<CENTER>"
            text+= "<nobr>"
            text+= '<table width="80%" border="0" cellspacing="1" cellpadding="1" >'
            text+= "<tr>"
            for l in range(len(labels)):
                if l < 2:
                    text+= '<td align="left" bgcolor=%s><b>%s</b></td>' % (hcolor, labels[l])
                elif l == 2:
                    text+= '<td align="center" bgcolor=%s><b>%s</b></td>' % (hcolor, labels[l])
                else:
                    text+= '<td align="right" bgcolor=%s><b>%s</b></td>' % (hcolor, labels[l])
            text+= "</tr>"
            line = 0
            for group in result['groups']:
                text+=("<tr>")
                element,group0 = string.split(group)
                fitarea    = "%.6e" % result['fitarea'][group]
                sigmaarea  = "%.2e" % result['sigmaarea'][group]
                area       = "%.6e" % result['area'][group]
                if mmolarflag:
                    fraction   = "%.4g" % result['mmolar'][group]
                else:
                    fraction   = "%.4g" % result['mass fraction'][group]
                if 'Expected Area' in labels:
                    fields = [element,group0,fitarea,sigmaarea,area,fraction]
                else:
                    fields = [element,group0,fitarea,sigmaarea,fraction]
                if result.has_key('layerlist'):
                    for layer in result['layerlist']:
                        if result[layer]['mass fraction'][group] < 0.0:
                            fraction   = "Unknown"
                        else:
                            if mmolarflag:
                                fraction   = "%.4g" % result[layer]['mmolar'][group]
                            else:
                                fraction   = "%.4g" % result[layer]['mass fraction'][group]
                        fields += [fraction]
                if line % 2:
                    color = lemmon
                else:
                    color = white
                i = 0 
                for field in fields:
                    if (i<2):
                        #text += '<td align="left"  bgcolor="%s"><b>%s</b></td>' % (color, field)
                        text += '<td align="left"  bgcolor=%s>%s</td>' % (color, field)
                    else:
                        #text += '<td align="right" bgcolor="%s"><b>%s</b></td>' % (color, field)
                        text += '<td align="right" bgcolor=%s>%s</td>' % (color, field)
                    i+=1
                text += '</tr>'
                line +=1           
            text+=("</table>")
            text+=("</nobr>")
            text+="</CENTER>"
        return text        

    def getConcentrationsAsAscii(self, concentrations=None):
        text = ""
        if concentrations is None:return text
        result =concentrations       
        #the table
        if result.has_key('mmolar'):
            mmolarflaglist = [False, True]
        else:
            mmolarflaglist = [False]
        for mmolarflag in mmolarflaglist:
            if mmolarflag:
                labels = ['Element','Group','Fit_Area','Sigma_Area', 'mM_Concentration']
            else:
                labels = ['Element','Group','Fit_Area','Sigma_Area', 'Mass_fraction']
            if result.has_key('layerlist'):
                if type(result['layerlist']) != type([]):
                    result['layerlist'] = [result['layerlist']]
                for label in result['layerlist']:
                    labels += [label.replace(' ','')]
            for l in labels:
                text+="%s  " % l
            text+=("\n")
            line = 0
            for group in result['groups']:
                element,group0 = string.split(group)
                fitarea    = "%.6e" % result['fitarea'][group]
                sigmaarea  = "%.2e" % result['sigmaarea'][group]
                area       = "%.6e" % result['area'][group]
                if mmolarflag:
                    fraction   = "%.4g" % result['mmolar'][group]
                else:
                    fraction   = "%.4g" % result['mass fraction'][group]
                if 'Expected Area' in labels:
                    fields = [element,group0,fitarea,sigmaarea,area,fraction]
                else:
                    fields = [element,group0,fitarea,sigmaarea,fraction]
                if result.has_key('layerlist'):
                    for layer in result['layerlist']:
                        if result[layer]['mass fraction'][group] < 0.0:
                            fraction   = "Unknown"
                        else:
                            if mmolarflag:
                                fraction   = "%.4g" % result[layer]['mmolar'][group]
                            else:
                                fraction   = "%.4g" % result[layer]['mass fraction'][group]
                        fields += [fraction]
                i = 0 
                for field in fields:
                    text += '%s  ' % (field)
                    i+=1
                text += '\n'
                line +=1
        return text

class ConcentrationsTool:
    def __init__(self, config = None, fitresult=None):
        self.config = {}
        self.config ['usematrix'] = 0
        self.config ['useattenuators'] = 1
        self.config ['usemultilayersecondary'] = 0
        self.config ['flux'] = 1.0E10
        self.config ['time'] = 1.0
        self.config ['area'] = 30.0
        self.config ['distance'] = 10.0
        self.config ['reference'] = "Auto"
        self.config ['mmolarflag'] = 0
        if config is not None: self.configure(config)
        self.fitresult = fitresult
    
    
    def configure(self, ddict = None):
        if ddict is None: ddict ={}
        for key in ddict:
            if key in self.config.keys():
                self.config[key] = ddict[key]
        return copy.deepcopy(self.config)
        
    def processFitResult(self, config = None, fitresult=None,
                         elementsfrommatrix = False, fluorates = None):
        #I should check if fit was successful ...        
        if fitresult is None: fitresult = self.fitresult
        else: self.fitresult = fitresult
        if config is None: config = self.config
        else:self.config=config
        if not self.config.has_key('usemultilayersecondary'):
            self.config['usemultilayersecondary']= 0
        secondary = self.config['usemultilayersecondary']
        #get attenuators and matrix from fit
        attenuators = []
        beamfilters = []
        funnyfilters = []
        matrix = None
        detectoratt = None
        multilayer = None
        for attenuator in fitresult['result']['config']['attenuators'].keys():
            if not fitresult['result']['config']['attenuators'][attenuator][0]: continue
            if attenuator.upper() == "MATRIX":
                matrix = fitresult['result']['config']['attenuators'][attenuator][1:4]
                alphain  = fitresult['result']['config']['attenuators'][attenuator][4]
                alphaout = fitresult['result']['config']['attenuators'][attenuator][5]
            elif attenuator.upper()[:-1] == "BEAMFILTER":
                beamfilters.append(fitresult['result']['config']['attenuators']\
                                                                [attenuator][1:])
            elif attenuator.upper() == "DETECTOR":
                detectoratt = fitresult['result']['config']['attenuators'][attenuator][1:]
            else:
                if len(fitresult['result']['config']['attenuators'][attenuator][1:]) == 4:
                   fitresult['result']['config']['attenuators'][attenuator].append(1.0)                
                if abs(fitresult['result']['config']['attenuators'][attenuator][4]-1.0) > 1.0e-10:
                    #funny attenuator
                    funnyfilters.append(fitresult['result']['config']['attenuators']\
                                                                [attenuator][1:])
                else:
                    attenuators.append(fitresult['result']['config']['attenuators']\
                                                                [attenuator][1:])
        if matrix is None:
            raise ValueError("Invalid or undefined sample matrix")
        
        if matrix[0].upper() == "MULTILAYER":
            layerlist = fitresult['result']['config']['multilayer'].keys()
            layerlist.sort()
            for layer in layerlist:
                if fitresult['result']['config']['multilayer'][layer][0]:
                    if multilayer is None:multilayer=[]
                    multilayer.append(fitresult['result']['config']['multilayer'][layer][1:])
                    if not Elements.isValidMaterial(multilayer[-1][0]):
                        raise ValueError("Material %s is not defined" % multilayer[-1][0])
        
        else:
            layerlist = ["Layer0"]
            multilayer= [matrix]
            if not Elements.isValidMaterial(matrix[0]):
                raise ValueError("Material %s is not defined" % matrix[0])
        energyList = fitresult['result']['config']['fit']['energy']
        if energyList is None:
            raise ValueError("Invalid energy")
        if type(energyList) != types.ListType:
            energyList    = [energyList]
            flagList   = [1]
            weightList = [1.0]
        else:
            flagList   = fitresult['result']['config']['fit']['energyflag']
            weightList = fitresult['result']['config']['fit']['energyweight']
        maxenergy = None
        finalEnergy = []
        finalWeight =[]
        finalFlag   = []
        for idx in range(len(energyList)):
            if flagList[idx]:
                energy = energyList[idx]
                if energy is None:
                    raise ValueError(\
                          "Energy %d isn't a valid energy" % idx)
                if energy <= 0.001:
                    raise ValueError(\
                          "Energy %d with value %f isn't a valid energy" %\
                          (idx, energy))
                if weightList[idx] is None:
                    raise ValueError(\
                          "Weight %d isn't a valid weight" % idx)
                if weightList [idx] < 0.0:
                    raise ValueError(\
                          "Weight %d with value %f isn't a valid weight" %\
                          (idx, weightList[idx]))
                finalEnergy.append(energy)
                finalWeight.append(weightList[idx])
                finalFlag.append(1)
        totalWeight = sum(weightList)
        if totalWeight == 0.0:
            raise ValueError("Sum of energy weights is 0.0")
        weightList = [x/totalWeight for x in finalWeight]
        energyList = finalEnergy
        flagList   = finalFlag

        #get elements list from fit, not from matrix
        groupsList = fitresult['result']['groups'] * 1
        if type(groupsList) != types.ListType:
            groupsList = [groupsList]
        
        todelete = []
        for i in range(len(groupsList)):
            ele = groupsList[i].split()[0]
            if len(ele) >2:
                todelete.append(i)
        if len(todelete):
            todelete.reverse()
            for i in todelete:
                del groupsList[i]
        
        elements = []
        newelements = []
        for group in groupsList:
            splitted = group.split()
            ele = splitted[0]
            newelements.append([Elements.getz(splitted[0]),
                                splitted[0],splitted[1]])
            if len(elements):
                if elements[-1] != ele:
                    elements.append(ele)
            else:
                elements.append(ele)
        newelements.sort()
        elements.sort()
        if not config['useattenuators']:
            attenuators  = None
            funnyfilters = None
        #import time
        #t0=time.time()
        if elementsfrommatrix:
            newelementsList = []
            for ilayer in range(len(multilayer)):
                pseudomatrix = multilayer[ilayer]
                eleDict = Elements.getMaterialMassFractions([pseudomatrix[0]], [1.0])
                if eleDict == {}:
                    raise ValueError(\
                        "Invalid layer material %s" % pseudomatrix[0])
                keys = eleDict.keys()
                for ele in keys:
                    for group in newelements:
                        if ele == group[1]:
                            if not group in newelementsList:
                                newelementsList.append(group)
            newelementsList.sort()
            fluo0 = Elements.getMultilayerFluorescence(multilayer,
                         energyList,
                         layerList = None,
                         weightList=weightList,
                         flagList=weightList,
                         fulloutput=1,
                         beamfilters=beamfilters * 1,
                         attenuators=attenuators * 1,
                         elementsList = newelementsList * 1,
                         alphain = alphain,
                         alphaout = alphaout,
                         cascade = True,
                         detector = detectoratt,
                         funnyfilters = funnyfilters * 1,
                         forcepresent=0,
                         secondary=secondary)
            fluototal = fluo0[0]
            fluolist  = fluo0[1:]
        else:
            if matrix[0].upper() != "MULTILAYER":
                multilayer = [matrix * 1]
            if fluorates is None:
                fluo0 = Elements.getMultilayerFluorescence(multilayer,
                             energyList,
                             layerList = None,
                             weightList=weightList,
                             flagList=flagList,
                             fulloutput=1,
                             beamfilters=beamfilters * 1,
                             attenuators=attenuators * 1,
                             elementsList = newelements * 1,
                             alphain = alphain,
                             alphaout = alphaout,
                             cascade = True,
                             detector=detectoratt,
                             funnyfilters=funnyfilters * 1,
                             forcepresent=1,
                             secondary=secondary)
            else:
                fluo0 = fluorates
            fluototal = fluo0[0]
            fluolist  = fluo0[1:]
        #I'll need total fluo element by element at some point
        #print "getMatrixFluorescence elapsed = ",time.time()-t0
        if config['usematrix']:
            present = []
            referenceLayerDict = {}
            materialComposition = []
            for ilayer in range(len(multilayer)):
                pseudomatrix = multilayer[ilayer] * 1
                #get elemental composition from matrix
                materialComposition.append(Elements.getMaterialMassFractions([pseudomatrix[0]], [1.0]))
                keys = materialComposition[-1].keys()
                materialElements = [[Elements.getz(x),x] for x in keys]
                materialElements.sort()
                for z,key in materialElements:
                    for ele in elements:
                        if key == ele:
                            present.append(key)
                            if not referenceLayerDict.has_key(ele):referenceLayerDict[ele] = []
                            referenceLayerDict[ele].append(ilayer)                                
            if len(present) == 0:
                text  = "Matrix must contain at least one fitted element\n"
                text += "in order to estimate flux and efficiency from it."
                raise ValueError(text)
            referenceElement = config['reference'].replace(' ',"")
            if len(referenceElement) and (referenceElement.upper() != 'AUTO'):
                if Elements.isValidFormula(referenceElement):
                    if len(referenceElement) == 2:                    
                        referenceElement = referenceElement.upper()[0] +\
                                           referenceElement.lower()[1]
                    elif len(referenceElement) == 1:
                        referenceElement = referenceElement.upper()[0]
                    if not (referenceElement in elements):
                        text = "Element %s not among fitted elements" % referenceElement
                        raise ValueError(text)
                    elif not (referenceElement in present):
                        text = "Element %s not among matrix elements" % referenceElement
                        raise ValueError(text)
                    referenceLayers  = referenceLayerDict[referenceElement]
                else:
                    text = "Element %s not a valid element" % referenceElement
                    raise ValueError(text)
            elif len(present) == 1:
                referenceElement = present[0]
                referenceLayers  = referenceLayerDict[referenceElement]
            else:
                #how to choose? Best fitted, largest fit area or greater concentration?
                #or better to give a weight to the different shells, energies , ...?
                referenceElement = present[0]
                fom = self._figureOfMerit(present[0],fluototal,fitresult)
                for key in present:
                    #if materialComposition[key] > materialComposition[referenceElement]:
                    #    referenceElement = key
                    newfom = self._figureOfMerit(key,fluototal,fitresult)
                    if newfom > fom:
                        fom =  newfom
                        referenceElement = key
                referenceLayers  = referenceLayerDict[referenceElement]
            solidangle    = 1.0
            for group in groupsList:
                item = group.split()
                element = item[0]
                transitions = item[1] + " xrays"
                if element == referenceElement:
                    break
            theoretical = 0.0
            for ilayer in referenceLayers:
                if elementsfrommatrix:
                    theoretical += fluolist[ilayer][referenceElement]['rates'][transitions] * \
                                   fluolist[ilayer][referenceElement]['mass fraction']
                else:
                    theoretical  += materialComposition[ilayer][referenceElement] * \
                                    fluolist[ilayer][referenceElement]['rates'][transitions]
            if theoretical <= 0.0:
                raise ValueError(\
                    "Theoretical rate is almost 0.0 Impossible to determine flux")
            else:
                flux = fitresult['result'][group]['fitarea'] / theoretical
        else:
            #solidangle = config['area']/(4.0 * Numeric.pi * pow(config['distance'],2))
            radius2 = config['area']/Numeric.pi
            solidangle = 0.5 * (1.0 -  (config['distance']/Numeric.sqrt(pow(config['distance'],2)+ radius2)))
            flux       = config['flux'] * config['time']
        
        #print "OBTAINED FLUX * SOLID ANGLE= ",flux * solidangle
        #print "flux * time = ",flux
        #print "actual solid angle = ",0.5 * (1.0 -  (config['distance']/Numeric.sqrt(pow(config['distance'],2)+ config['area']/Numeric.pi)))
        #print "solid angle factor= ",solidangle
        #ele  = 'Pb'
        #rays = "L xrays"
        #print "theoretical = ",fluototal[ele]['rates'][rays]
        #print "expected    = ",flux * solidangle * fluototal[ele]['rates'][rays]
        
        #for ilayer in range(len(multilayer)):
        #    print "ilayer = ",ilayer, "theoretical = ",fluolist[ilayer][ele]['rates'][rays]
        #    print "ilayer = ",ilayer, "expected = ",flux * solidangle * fluolist[ilayer][ele]['rates'][rays]
        dict = {}
        dict['groups'] = groupsList
        dict['elements'] = elements
        dict['mass fraction'] = {}
        if config.has_key('mmolarflag'):
            if config['mmolarflag']:
                dict['mmolar'] = {}
        else:
            config['mmolarflag'] = 0
        dict['area'] = {}
        dict['fitarea'] = {}
        dict['sigmaarea'] = {}
        fluo = fluototal

        for group in groupsList:
            item = group.split()
            element = item[0]
            transitions = item[1] + " xrays"
            if element in fluo.keys():
                if fluo[element].has_key(transitions):
                    #this SHOULD be with concentration one
                    theoretical = fluo[element]['rates'][transitions] * 1.0
                    expected = theoretical * flux * solidangle
                    concentration = fitresult['result'][group]['fitarea']/expected
                else:
                    theoretical   = 0.0
                    concentration = 0.0
            else:
                theoretical   = 0.0
                concentration = 0.0
            #dict['area'][group]    = theoretical * flux * solidangle * concentration
            dict['fitarea'][group] = fitresult['result'][group]['fitarea']
            dict['sigmaarea'][group] = fitresult['result'][group]['sigmaarea']
            if elementsfrommatrix:
                if element in fluo.keys():
                    dict['mass fraction'][group] = 1.0 * fluo[element]['mass fraction']
                else:
                    dict['mass fraction'][group] = 0.0
                dict['area'][group]    = theoretical * flux * solidangle *\
                                         dict['mass fraction'][group]
            else:
                dict['mass fraction'][group] = concentration
                dict['area'][group]    = theoretical * flux * solidangle
            if config['mmolarflag']:
                #mM = (mass_fraction * density)/atomic_weight
                dict['mmolar'] [group]= 1000000. *\
                                        (multilayer[0][1] * dict['mass fraction'][group])/Elements.Element[element]['mass']

        #I have the globals/average values now I calculate layer per layer
        #if necessary
        dict['layerlist'] = []
        if matrix[0].upper() == "MULTILAYER":
            ilayer = 0
            for layer in layerlist:
                if fitresult['result']['config']['multilayer'][layer][0]:
                    dict['layerlist'].append(layer)
                    dict[layer] = {}
                    dict2 = dict[layer]
                    dict2['groups'] = groupsList
                    dict2['elements'] = elements
                    dict2['mass fraction'] = {}
                    if config['mmolarflag']:
                        dict2['mmolar'] = {}
                    dict2['area'] = {}
                    dict2['fitarea'] = {}
                    fluo =fluolist[ilayer]
                    for group in groupsList:
                        item = group.split()
                        element = item[0]
                        transitions = item[1] + " xrays"
                        if element in fluo.keys():
                            if fluo[element].has_key(transitions):
                                theoretical = fluo[element]['rates'][transitions] * 1
                                expected = theoretical * flux * solidangle
                                if expected > 0.0:
                                    concentration = fitresult['result'][group]['fitarea']/expected
                                else:
                                    concentration = -1
                            else:
                                theoretical   = 0.0
                                concentration = 0.0
                        else:
                            theoretical   = 0.0
                            concentration = 0.0
                        dict2['fitarea'][group] = 1 * fitresult['result'][group]['fitarea']
                        if elementsfrommatrix:
                            if element in fluo.keys():
                                dict2['mass fraction'][group] = 1 * fluo[element]['mass fraction']
                            else:
                                dict2['mass fraction'][group] = 0.0
                            #I calculate matrix in optimized form, 
                            #so I have to multiply by the mass fraction
                            dict2['area'][group]    = theoretical * flux * solidangle *\
                                                      dict2['mass fraction'][group]
                        else:
                            dict2['mass fraction'][group] = concentration
                            dict2['area'][group]    = theoretical * flux * solidangle
                        if config['mmolarflag']:
                            #mM = (mass_fraction * density)/atomic_weight
                            dict2['mmolar'] [group]= 1000000. * (multilayer[ilayer][1] * \
                                                      dict2['mass fraction'][group])/Elements.Element[element]['mass']
                        #if group == "Pb L":
                        #    print "layer", ilayer,'area ', dict2['area'][group]
                        #    print "layer", ilayer,'mass fraction =', dict2['mass fraction'][group]
                    ilayer += 1
            if elementsfrommatrix:
                for group in groupsList:
                    dict['area'][group] = 0.0
                    for layer in dict['layerlist']:
                        if group in dict[layer]['area'].keys():
                            dict['area'][group] += dict[layer]['area'][group]

        return dict
    

    def _figureOfMerit(self,element,fluo,fitresult):
        weight = 0.0
        for transitions in fluo[element]['rates'].keys():
            if fluo[element]['rates'][transitions] > 0.0:
                if   (transitions[0] == "K") and (Elements.getz(element) > 18):
                    factor = 2.0 
                elif (transitions[0] == "L") and (Elements.getz(element) > 54):
                    factor = 1.5
                else:
                    factor = 1.0
                group = element+" "+transitions.split()[0]
                if group in fitresult['result']['groups']:
                    fitarea = fitresult['result'][group]['fitarea']
                    weightHelp = fitarea * fluo[element]['rates'][transitions] * factor * \
                                fluo[element]['mass fraction']
                    if weightHelp > weight:
                        weight = weightHelp
        return weight        
                    
if __name__ == "__main__":
    import sys
    import ConfigDict
    import getopt
    if len(sys.argv) > 1:
        options = ''
        longoptions = ['flux=','time=','area=','distance=','attenuators=','usematrix=']
        tool = ConcentrationsTool()
        opts, args = getopt.getopt(
                        sys.argv[1:],
                        options,
                        longoptions)
        config = tool.configure()
        for opt,arg in opts:
            if opt in ('--flux'):
                config['flux'] = float(arg)
            elif opt in ('--area'):
                config['area'] = float(arg)
            elif opt in ('--time'):
                config['time'] = float(arg)
            elif opt in ('--distance'):
                config['distance'] = float(arg)
            elif opt in ('--attenuators'):
                config['useattenuators'] = int(float(arg))
            elif opt in ('--usematrix'):
                config['usematrix'] = int(float(arg))
        tool.configure(config) 
        filelist = args
        for file in filelist:
            d = ConfigDict.ConfigDict()
            d.read(file)
            for material in d['result']['config']['materials'].keys():
                Elements.Material[material] = copy.deepcopy(d['result']['config']['materials'][material])
            print(tool.processFitResult(fitresult=d, elementsfrommatrix=True))
    else:
        print("Usage:")
        print("ConcentrationsTool [--flux=xxxx --area=xxxx] fitresultfile")

