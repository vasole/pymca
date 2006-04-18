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
"""
    SPSData.py
    Data derived class to access spec shared memory
"""

#from PyDVT import __version__,__date__,__author__


################################################################################  
#from Data import *
import spswrap as sps
################################################################################

SOURCE_TYPE = "SPS"    


class SPSLayer:
    """
    Specializes Data class to access Spec shared memory.
    Interface: Data class interface.
    """
    def __init__(self,refresh_interval=None,info={}):
        """
        See Data.__init__
        """
        self.EdfObj=None
        info["Class"]="SPSData"
        #Data.__init__(self,refresh_interval,info)        
        self.SourceName= None
        self.SourceInfo= None
        self.GetData   = self.LoadSource

    def GetPageInfo(self,index={}):
        if index.has_key('SourceName'):
            self.SetSource(index['SourceName'])
            if index.has_key('Key'):
                info=self.GetData(index['Key'])   
                return info[0]

    def AppendPage(self,scan_info, scan_data):
        return scan_info,scan_data

        
    def SetSource (self,source_name=None):
        """
        Sets a new source for data retrieving, an spec version.
        If spec exists, self.Source will be this spec name.
        Parameters:
        source_name: name of spec version
        """
        if source_name==self.SourceName: return 1
        
        if (source_name != None) and (source_name in sps.getspeclist()):
            self.SourceName=source_name
            self.Source=self.SourceName
            return 1
        else:
            self.SourceName=None
            self.Source=None
            return 0


    def GetSourceInfo (self, key=None):
        """
        Returns information about the Spec version set by
        SetSource, to give application possibility to know about
        it before loading.
        Returns a dictionary with the keys "Size" (number of possible
        keys to this source) and "KeyList" (list of all available keys
        in this source). Each element in "KeyList" is an shared memory
        array name.
        If key is set as an array name, returns information about it.
        """        
        if self.SourceName is not None: 
            if key is None: return self.__GetSourceInfo()
            elif key in sps.getarraylist(self.SourceName): return self.__GetArrayInfo(key)
        return None

    def __GetSourceInfo(self):
        arraylist= []
        for array in sps.getarraylist(self.SourceName):
	    arrayinfo= sps.getarrayinfo(self.SourceName, array)
	    arraytype= arrayinfo[2]
	    arrayflag= arrayinfo[3]
            if arrayflag in (sps.IS_ARRAY, sps.IS_MCA, sps.IS_IMAGE) and arraytype!=sps.STRING:
                    arraylist.append(array)
        source_info={}
        source_info["Size"]=len(arraylist)
        source_info["KeyList"]=arraylist
        return source_info


    def __GetArrayInfo(self,array):
        info={}
        info["SourceType"]=SOURCE_TYPE
        info["SourceName"]=self.SourceName
        info["Key"]=array
        info["Source"]=self.Source
        
        arrayinfo=sps.getarrayinfo (self.SourceName,array)
        info["rows"]=arrayinfo[0]
        info["cols"]=arrayinfo[1]
        info["type"]=arrayinfo[2]
        info["flag"]=arrayinfo[3]
        counter=sps.updatecounter (self.SourceName,array)
        info["updatecounter"]=counter

        envdict={}
        keylist=sps.getkeylist (self.SourceName,array+"_ENV")
        for i in keylist:
            val=sps.getenv(self.SourceName,array+"_ENV",i)
            envdict[i]=val
        info["envdict"]=envdict

        calibarray= array + "_PARAM"
        if calibarray in sps.getarraylist(self.SourceName):
            try:
                data= sps.getdata(self.SourceName, calibarray)
                updc= sps.updatecounter(self.SourceName, calibarray)
                info["EnvKey"]= calibarray
                info["McaCalib"]= data.tolist()[0]
                info["env_updatecounter"]= updc
            except:
                pass

        if array in ["XIA_DATA", "XIA_BASELINE"]:
            envarray= "XIA_DET"
            if envarray in sps.getarraylist(self.SourceName):
                try:
                    data= sps.getdata(self.SourceName, envarray)
                    updc= sps.updatecounter(self.SourceName, envarray)
                    info["EnvKey"]= envarray
                    info["Detectors"]= data.tolist()[0]
                    info["env_updatecounter"]= updc
                except:
                    pass
	
        return info

 
    def LoadSource(self,key_list="ALL",append=0,invalidate=1,row="ALL",col="ALL"):
        """
        Creates a given number of pages, getting data from the actual
        source (set by SetSource)
        Parameters:
        key_list: list of all keys to be read from source. It is a list of
                 string, shared memory array names, to be read from the file.
                 It can be also one single string, if only one array is to be read.
        append: If non-zero appends to the end of page list.
                Otherwise, initializes the page list                
        invalidate: if non-zero performas an invalidade call after
                    loading
        row: If set to an integer, loads a single row (0-based indexed)
        col: If set to an integer, loads a single column (0-based indexed)
        """
        #AS if append==0: Data.Delete(self)        
        if type(key_list) == type(" "): key_list=(key_list,)
        output =[]
        if self.SourceName in sps.getspeclist():
            if key_list == "ALL": key_list=sps.getarraylist(self.SourceName)
            for array in key_list:
                if array in sps.getarraylist(self.SourceName):
                    info=self.__GetArrayInfo(array)
                    info["row"]=row
                    info["col"]=col
                    if info["row"]!="ALL":
                        data= sps.getdatarow(self.SourceName,array,info["row"])
                        if data is not None: data=Numeric.reshape(data,(1,data.shape[0]))
                    elif info["col"]!="ALL":
                        data= sps.getdatacol(self.SourceName,array,info["col"])
                        if data is not None: data=Numeric.reshape(data,(data.shape[0],1))
                    else: data=sps.getdata (self.SourceName,array)
                    #self.AppendPage(info,data)
                    output.append([info,data])
        if len(output) == 1:
            return output[0]
        else:
            return output

        #AS if invalidate: self.Invalidate()

 
    def __RefreshPageOrig (source_obj,self,page):
        """
        Virtual method, implements seeking for changes in data.
        Returns non-zero if the page was changed.
        If not implemented in the derived class, this class doesn't
        support dinamic changes monitoring.
        As pages can be copied to different Data objects, and can
        store the original RefreshPage method for updating, source_obj
        refers to the object that was origin of the page data, while
        self indicates the object that actually owns the page
        with index page.
        It was done this way because if it is stored the reference to
        the unbound method, python doesn't allow you to call it with
        an object of different data type.

        Important:
        Derived classes shall update the page:   self.Pages[page]
        but not:   source_obj.Pages[page]
        """       
        if (self.GetItemPageInfo("SourceType",page)==SOURCE_TYPE):
            specname=self.GetItemPageInfo("SourceName",page)
            arrayname=self.GetItemPageInfo("Key",page)
            updatecounter=self.GetItemPageInfo("updatecounter",page)
            if (updatecounter!=None) and (arrayname!=None) and (specname!=None):
                if specname in sps.getspeclist():
                    if arrayname in sps.getarraylist(specname):
                        counter=sps.updatecounter (specname,arrayname)
                        if (counter != updatecounter):
                            info=self.GetPageInfo(page)
                            info.update(self.__GetArrayInfo(arrayname))
                            if   info["row"]!="ALL": data= sps.getdatarow(specname,arrayname,info["row"])
                            elif info["col"]!="ALL": data= sps.getdatacol(specname,arrayname,info["col"])
                            else: data=sps.getdata (specname,arrayname)                            
                            self.Pages[page].Array=data
                            return 1
	    infoname= self.GetItemPageInfo("EnvKey")
	    infoupdc= self.GetItemPageInfo("env_updatecounter")
	    if (infoupdc!=None) and (infoname!=None) and (specname!=None):
		if infoname in sps.getarraylist(specname):
			counter= sps.updatecounter(specname,infoname)
			if (counter!=infoupdc):
				info= self.GetPageInfo(page)
				info.update(self.__GetArrayInfo(arrayname))
				return 1
        return 0

    def RefreshPage(self,sourcename,key):
        specname = sourcename
        arrayname= key
        if not sps.specrunning(specname):
            return 0
        
        if sps.isupdated(specname,arrayname):
            return 1
        else:
            return 0

################################################################################
#EXEMPLE CODE:
    
if __name__ == "__main__":
    import sys,time

    try:
        obj=SPSLayer()
        specname=sys.argv[1]
        arrayname=sys.argv[2]        
        obj.SetSource(specname)
        obj.LoadSource(arrayname)
        while(1):
            time.sleep(1)
            print obj.RefreshPage(specname,arrayname)
    except:
        print "Usage: SPSLayer.py <specversion> <specname>"
        sys.exit()

    for i in range (obj.GetNumberPages()):
        print obj.GetPageInfo(i)
        print obj.GetPageArray(i)

    

