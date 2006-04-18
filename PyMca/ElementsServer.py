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
from TacoServer  import *
import Numeric
import string

class ElementsServer(TacoServer):
    "This is a TacoServer class"
    try:
        #This constant should be imported in TacoServer
        ElementsBase = DevPythonBase
    except:
        print "Setting ElementsBase"
        ElementsBase = 348651520
    DevState                    = ElementsBase + 1
    GetMaterialID               = ElementsBase + 2
    GetFormula                  = ElementsBase + 3
    GetName                     = ElementsBase + 4
    GetDensity                  = ElementsBase + 5
    GetCrossSection             = ElementsBase + 6
    #GetMultipleData             = ElementsBase + 7
    #GetTransmission             = ElementsBase + 8
    #GetAbsorption               = ElementsBase + 9
    cmd_list = {DevState:       [D_VOID_TYPE,     D_SHORT_TYPE,  'getState', 'DevState'],
                GetMaterialID:  [D_STRING_TYPE,  D_FLOAT_TYPE,   'getMaterialID','GetMaterialID'],
                GetFormula:     [D_FLOAT_TYPE,    D_STRING_TYPE, 'getFormula','GetFormula'],
                GetName:        [D_FLOAT_TYPE,    D_STRING_TYPE, 'getName',   'GetName'],
                GetDensity:     [D_FLOAT_TYPE,    D_FLOAT_TYPE,  'getDensity','GetDensity'],
                GetCrossSection:[D_VAR_FLOATARR,    D_VAR_FLOATARR,'getCrossSection','GetCrossSection'],
                #GetMultipleData:[D_VAR_FLOATARR,    D_VAR_FLOATARR,'getMultipleData','GetMultipleData'],
                #GetTransmission:[D_VAR_FLOATARR,    D_VAR_FLOATARR,'getTransmission','GetTransmission'],
                #GetAbsorption:  [D_VAR_FLOATARR,    D_VAR_FLOATARR,'getAbsorption','GetAbsorption']
                }
    class_name = "ElementsServer TacoServer Class"
    def __init__(self,devicename):
        TacoServer.__init__ (self, devicename)
        self.__names   =[]
        self.__formules=[]
        self.__ids  ={}      
       
    def getState(self):
        return 1

    def getMaterialID(self,name):
        if name in self.__names:
            return self.__names.index(name)
        if name in self.__formules:
            return self.__formules.index(name)
        #one has to try to define a new element/compound
        #I should check if the name as it is exists in user and default directories
        if name in Elements.Element.keys():
            #single element
            self.__names.append(Elements.Element[name]['name'])
            self.__formules.append(name)
            key = self.__formules.index(name)
            self.__ids[key] = Elements.getmassattcoef(name)
            self.__ids[key]['density'] = Elements.Element[name]['density']
            self.__ids[key]['energy'] = Numeric.array(self.__ids[key]['energy'])
        else:
            #compound
            try:
                self.__names.append(name)
                self.__formules.append(name)
                matt= Elements.getmassattcoef(name)
                key = self.__formules.index(name)
                self.__ids[key] = matt
                if self.__ids[key] == {}:
                    del self.__ids[key]
                    del self.__formules[key]
                    del self.__names[key]
                    key = -1
                else:
                    self.__ids[key]['energy'] = Numeric.array(self.__ids[key]['energy'])
            except:
                key= -1
        return key
        
    def getFormula(self,key0):
        try:
           key=int(key0)
           if key in self.__ids.keys():
               return self.__formules[key]
           else:
               return ""    
        except:
            return -1
            
    def getName(self,key0):
        try:
            key=int(key0)
            if key in self.__ids.keys():
                return self.__names[key]
            else:
                return ""
        except:
            return -1
            
    def getDensity(self,key0):
        try:
            key=int(key0)
            if key in self.__ids.keys():
                if self.__ids[key].has_key('density'):
                    return self.__ids[key]['density']
                else:
                    return 0
        except:
            return -1
        
    def getCrossSection(self,x):    
        #x to be taken as an array
        #1st element is ID
        #rest are energies
        #associative arrays arrive as strings to python from SPEC
        try:
            pyarray=Numeric.array(map(string.atof,x))
            key = int(pyarray[0])
            if key in self.__ids.keys():
                if len(x) == 1:
                    return Numeric.concatenate((self.__ids[key]['energy'],
                                    self.__ids[key]['total'],
                                    self.__ids[key]['photo'],
                                    self.__ids[key]['coherent'],
                                    self.__ids[key]['compton'],
                                    self.__ids[key]['pair']),1)
                else:
                    energy = pyarray[1:]
                    xcom_data={}
                    xcom_data.update(self.__ids[key])
                    dict={}
                    dict['energy']=[]
                    dict['total']=[]
                    dict['photo']=[]
                    dict['coherent']=[]
                    dict['compton']=[]
                    dict['pair']=[]
                    for ene in energy:
                        i0=max(Numeric.nonzero((xcom_data['energy'] <= ene)))
                        i1=min(Numeric.nonzero((xcom_data['energy'] >= ene)))
                        if i1 == i0:
                            cohe=xcom_data['coherent'][i1]
                            comp=xcom_data['compton'][i1]
                            photo=xcom_data['photo'][i1]
                            pair=xcom_data['pair'][i1]
                        else:
                            A=xcom_data['energy'][i0]
                            B=xcom_data['energy'][i1]
                            c2=(ene-A)/(B-A)
                            c1=(B-ene)/(B-A)

                            cohe= pow(10.0,c2*Numeric.log10(xcom_data['coherent'][i1])+\
                                                    c1*Numeric.log10(xcom_data['coherent'][i0]))
                            comp= pow(10.0,c2*Numeric.log10(xcom_data['compton'][i1])+\
                                                    c1*Numeric.log10(xcom_data['compton'][i0]))
                            photo=pow(10.0,c2*Numeric.log10(xcom_data['photo'][i1])+\
                                                    c1*Numeric.log10(xcom_data['photo'][i0]))
                            if xcom_data['pair'][i1] > 0.0:
                                c2 = c2*Numeric.log10(xcom_data['pair'][i1])
                                if xcom_data['pair'][i0] > 0.0:
                                    c1 = c1*Numeric.log10(xcom_data['pair'][i0])
                                    pair = pow(10.0,c1+c2)
                                else:
                                    pair =0.0
                            else:
                                pair =0.0
                        dict['energy'].append(ene)
                        dict['coherent'].append(cohe)
                        dict['compton'].append(comp)
                        dict['photo'].append(photo)
                        dict['pair'].append(pair)
                        dict['total'].append((cohe+comp+photo+pair))
                    return Numeric.concatenate((dict['energy'],
                                                dict['total'],
                                                dict['photo'],
                                                dict['coherent'],
                                                dict['compton'],
                                                dict['pair']),1)


                    
            else:
                return [0]
        except:
            #error
            return [-1]
        
        
    def getMultipleData(self,key):
        return 
        
        
    def getAttenuation(self,indata):
        return
        
        
    def getTransmission(self,indata):
        return

if __name__ == "__main__":
        if len(sys.argv) > 1:
            name= sys.argv[1]
        else:
            name= "ElementsServer"
        if len(sys.argv) > 2:
            pn= int(sys.argv[2])
        else:
            pn = 5000005

        try:
            dev=ElementsServer(name)
            server_startup((dev,),nodb=1,pn=pn)
            import time
            while(1):
                time.sleep(1000)
        except KeyboardInterrupt:
            print "ElementsServer: Exit on KeyboardInterrupt"
            sys.exit
 
