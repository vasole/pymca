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
"""
    EdfFileData.py
    Data derived class to access Edf files
"""



################################################################################
import logging
#import fast_EdfFile as EdfFile
from PyMca5.PyMcaIO import EdfFile
################################################################################

_logger = logging.getLogger(__name__)
SOURCE_TYPE = "EdfFile"


class EdfFileLayer(object):
    """
    Specializes Data class in order to access Edf files.
    Interface: Data class interface.
    """
    def __init__(self,refresh_interval=None,info={},fastedf=None):
        """
        See Data.__init__
        """
        info["Class"]="EdfFileData"
        #Data.__init__(self,refresh_interval,info)
        self.SourceName= None
        self.SourceInfo= None
        if fastedf is None:fastedf =0
        self.fastedf = fastedf
        self.GetData   = self.LoadSource

    def GetPageInfo(self,index={}):
        if 'SourceName' in index:
            self.SetSource(index['SourceName'])
            if 'Key' in index:
                info=self.GetData(index['Key'])
                return info[0]

    def AppendPage(self,info={}, array=None):
        return info,array

    def SetSource (self,source_name=None, source_obj = None):
        """
        Sets a new source for data retrieving, an edf file.
        If the file exists, self.Source will be the EdfFile
        object associated to this file.
        Parameters:
        source_name: name of the edf file
        """
        if source_name==self.SourceName: return 1
        if (type(source_name) != type([])):source_name = [source_name]
        if (source_name is not None):
            if source_obj is not None:
                self.Source= source_obj
            else:
                if (type(source_name) == type([])):
                    _logger.debug("List of files")
                    self.Source=[]
                    for name in source_name:
                        try:
                            self.Source.append(EdfFile.EdfFile(name,fastedf=self.fastedf))
                        except:
                            # _logger.info("EdfFileLayer.SetSource: Error trying to read EDF file %s", name)
                            self.Source.append( None)
                else:
                    try:
                        self.Source = EdfFile.EdfFile(source_name, fastedf=self.fastedf)
                    except:
                        # _logger.info("EdfFileLayer.SetSource: Error trying to read EDF file")
                        self.Source=None
        else:
            self.Source=None
        self.SourceInfo= None
        if self.Source is None:
            self.SourceName= None
            return 0
        else:
            self.SourceName=""
            for name in source_name:
                if self.SourceName != "":self.SourceName+="|"
                self.SourceName+= name
            return 1




    def GetPageArray(self,index=0):
        """
        Returns page's data (NumPy array)
        Parameters:
        index: Either an integer meaning the sequencial position of the page
               or a dictionary that logically index the page based on keys of
               the page's Info dictionary.
        """
        index=self.GetPageListIndex(index)
        if index is None or index >= len(self.Pages): return None
        return self.Pages[index].Array


    def GetPageListIndex(self,index):
        """
        Converts a physical or logical index, into a physical one
        """
        try:
            index = int(index)
        except:
            pass
        if type(index) is not  type({}): return index
        for i in range(self.GetNumberPages()):
            found = 1
            for key in index.keys():
                if key not in self.Pages[i].Info.keys() or self.Pages[i].Info[key] != index[key]:
                    found=0
                    break
            if found: return i
        return None

    def GetSourceInfo (self,key=None):
        """
        Returns information about the EdfFile object created by
        SetSource, to give application possibility to know about
        it before loading.
        Returns a dictionary with the keys "Size" (number of possible
        keys to this source) and "KeyList" (list of all available keys
        in this source). Each element in "KeyList" is an integer
        meaning the index of the array in the file.
        """
        if self.SourceName == None: return None
        if type(self.Source) == type([]):
            enumtype = 1
        else:
            enumtype = 0
        if not enumtype:
            if key is None:
                    source_info={}
                    if self.SourceInfo is None:
                        NumImages=self.Source.GetNumImages()
                        self.SourceInfo={}
                        self.SourceInfo["Size"]=NumImages
                        self.SourceInfo["KeyList"]=range(NumImages)
                    source_info.update(self.SourceInfo)
                    return source_info
            else:
                NumImages=self.Source.GetNumImages()
                source_info={}
                source_info["Size"]=NumImages
                source_info["KeyList"]=range(NumImages)
            return source_info
        else:
            if key is None:
                source_info={}
                self.SourceInfo={}
                self.SourceInfo["Size"]   = 0
                self.SourceInfo["KeyList"]= []
                for source in self.Source:
                    NumImages=source.GetNumImages()
                    self.SourceInfo["Size"] += NumImages
                    for imagenumber in range(NumImages):
                        self.SourceInfo["KeyList"].append('%d.%d' % (self.Source.index(source)+1,
                                                                     imagenumber+1))
                    source_info.update(self.SourceInfo)
                return source_info
            else:
                try:
                    index,image = key.split(".")
                    index = int(index)-1
                    image = int(image)-1
                except:
                    _logger.error("Error trying to interpret key = %s", key)
                    return {}
                source = self.Source[index]
                NumImages=source.GetNumImages()
                source_info={}
                source_info["Size"]=NumImages
                source_info["KeyList"]=[]
                for imagenumber in range(NumImages):
                    source_info.append('%d.%d' % (index+1,imagenumber+1))
            return source_info


    def LoadSource(self,key_list="ALL",append=0,invalidate=1,pos=None,size=None):
        """
        Creates a given number of pages, getting data from the actual
        source (set by SetSource)
        Parameters:
        key_list: list of all keys to be read from source. It is a list of
                 keys, meaning the indexes to be read from the file.
                 It can be also one integer, if only one array is to be read.
        append: If non-zero appends to the end of page list.
                Otherwise, initializes the page list
        invalidate: if non-zero performas an invalidade call after
                    loading
        pos and size: (x), (x,y) or (x,y,z) tuples defining a roi
                      If not defined, takes full array
                      Stored in page's info
        """
        #AS if append==0: Data.Delete(self)
        #numimages=self.Source.GetNumImages()
        sourceinfo = self.GetSourceInfo()
        numimages=sourceinfo['Size']
        if key_list == "ALL": key_list=sourceinfo['KeyList']
        elif type(key_list) != type([]): key_list=[key_list]

        #AS elif type(key_list) is types.IntType: key_list=[key_list]
        if pos is not None:
            edf_pos=list(pos)
            for i in range(len(edf_pos)):
                if edf_pos[i]=="ALL":edf_pos[i]=0
        else: edf_pos=None

        if size is not None:
            edf_size=list(size)
            for i in range(len(edf_size)):
                if edf_size[i]=="ALL":edf_size[i]=0
        else: edf_size=None

        output = []
        for key0 in key_list:
            f = 1
            sumrequested = 0
            if type(key0) == type({}):
                if 'Key' in key0:
                    key = key0['Key']
            if type(key0) == type(''):
                if len(key0.split(".")) == 2:
                    f,i = key0.split(".")
                    f=int(f)
                    i=int(i)
                    if (i==0):
                        if f == 0:sumrequested=1
                        else:i=1
                    key = "%d.%d" % (f,i)
                else:
                    i=int(key0)
                    if i < len(sourceinfo['KeyList']):
                        key = sourceinfo['KeyList'][i]
                        f,i = key.split(".")
                        f=int(f)
                        i=int(i)
                    else:
                        key = "%d.%d" % (f,i)
            else:
                i = key0
                if i >= numimages:
                    raise IndexError("EdfFileData: index out of bounds")
                imgcount   =0
                f=0
                for source in self.Source:
                    f+=1
                    n = source.GetNumImages()
                    if i < (imgcount+n):
                        i = i - imgcount
                        break
                    imgcount += n
                i+=1
                key = "%d.%d" % (f,i)
            if key == "0.0":sumrequested=1
            info={}
            info["SourceType"]=SOURCE_TYPE
            info["SourceName"]=self.SourceName
            info["Key"]=key
            info["Source"]=self.Source
            info["pos"]=pos
            info["size"]=size
            if not sumrequested:
                info.update(self.Source[f-1].GetStaticHeader(i-1))
                info.update(self.Source[f-1].GetHeader(i-1))
                if info["DataType"]=="UnsignedShort":array=self.Source[f-1].GetData(i-1,"SignedLong",Pos=edf_pos,Size=edf_size)
                elif info["DataType"]=="UnsignedLong":array=self.Source[f-1].GetData(i-1,"DoubleValue",Pos=edf_pos,Size=edf_size)
                else: array=self.Source[f-1].GetData(i-1,Pos=edf_pos,Size=edf_size)
                if 'Channel0' in info:
                    info['Channel0'] = int(float(info['Channel0']))
                elif 'MCA start ch' in info:
                    info['Channel0'] = int(float(info['MCA start ch']))
                else:
                    info['Channel0'] = 0
                if not ('McaCalib' in info):
                    if 'MCA a' in info and 'MCA b' in info and 'MCA c' in info:
                        info['McaCalib'] = [float(info['MCA a']),
                                            float(info['MCA b']),
                                            float(info['MCA c'])]
            else:
                # this is not correct, I assume info
                # is the same for all sources
                f=1
                i=1
                info.update(self.Source[f-1].GetStaticHeader(i-1))
                info.update(self.Source[f-1].GetHeader(i-1))
                if 'Channel0' in info:
                    info['Channel0'] = int(float(info['Channel0']))
                elif 'MCA start ch' in info:
                    info['Channel0'] = int(float(info['MCA start ch']))
                else:
                    info['Channel0'] = 0
                if not ('McaCalib' in info):
                    if 'MCA a' in info and 'MCA b' in info and 'MCA c' in info:
                        info['McaCalib'] = [float(info['MCA a']),
                                            float(info['MCA b']),
                                            float(info['MCA c'])]
                f=0
                for source in self.Source:
                    for i in range(source.GetNumImages()):
                        if info["DataType"]=="UnsignedShort":array0=source.GetData(i,"SignedLong",Pos=edf_pos,
                                                                                        Size=edf_size)
                        elif info["DataType"]=="UnsignedLong":array0=source.GetData(i,"DoubleValue",Pos=edf_pos,
                                                                                        Size=edf_size)
                        else: array0=source.GetData(i,Pos=edf_pos,Size=edf_size)
                        if (f==0) and (i==0):
                            array = 1 * array0
                        else:
                            array += array0
                    f+=1
            if 'McaCalib' in info:
                if type(info['McaCalib']) == type(" "):
                    info['McaCalib'] = info['McaCalib'].replace("[","")
                    info['McaCalib'] = info['McaCalib'].replace("]","")
                    cala, calb, calc = info['McaCalib'].split(",")
                    info['McaCalib'] = [float(cala),
                                        float(calb),
                                        float(calc)]

            output.append([info,array])
            #AS self.AppendPage(info,array)
        if len(output) == 1:
            return output[0]
        else:
            return output
        #AS if invalidate: self.Invalidate()

    def LoadSourceSingle(self,key_list="ALL",append=0,invalidate=1,pos=None,size=None):
        """
        Creates a given number of pages, getting data from the actual
        source (set by SetSource)
        Parameters:
        key_list: list of all keys to be read from source. It is a list of
                 keys, meaning the indexes to be read from the file.
                 It can be also one integer, if only one array is to be read.
        append: If non-zero appends to the end of page list.
                Otherwise, initializes the page list
        invalidate: if non-zero performas an invalidade call after
                    loading
        pos and size: (x), (x,y) or (x,y,z) tuples defining a roi
                      If not defined, takes full array
                      Stored in page's info
        """
        #AS if append==0: Data.Delete(self)
        numimages=self.Source.GetNumImages()
        if key_list == "ALL":
            key_list=range(numimages)
        elif type(key_list) != type([]): key_list=[key_list]
        #AS elif type(key_list) is types.IntType: key_list=[key_list]
        if pos is not None:
            edf_pos=list(pos)
            for i in range(len(edf_pos)):
                if edf_pos[i]=="ALL":edf_pos[i]=0
        else: edf_pos=None

        if size is not None:
            edf_size=list(size)
            for i in range(len(edf_size)):
                if edf_size[i]=="ALL":edf_size[i]=0
        else: edf_size=None

        output = []
        for key in key_list:
            if type(key) == type({}):
                if 'Key' in key:
                    key = key['Key']
            if type(key) == type(''):
                i=int(key)
            else:
                i = key
            if i >= numimages:
                raise IndexError("EdfFileData: index out of bounds")
            info={}
            info["SourceType"]=SOURCE_TYPE
            info["SourceName"]=self.SourceName
            info["Key"]=i
            info["Source"]=self.Source
            info["pos"]=pos
            info["size"]=size
            info.update(self.Source.GetStaticHeader(i))
            info.update(self.Source.GetHeader(i))
            if info["DataType"]=="UnsignedShort":array=self.Source.GetData(i,"SignedLong",Pos=edf_pos,Size=edf_size)
            elif info["DataType"]=="UnsignedLong":array=self.Source.GetData(i,"DoubleValue",Pos=edf_pos,Size=edf_size)
            else: array=self.Source.GetData(i,Pos=edf_pos,Size=edf_size)
            if 'MCA start ch' in info:
                info['Channel0'] = int(info['MCA start ch'])
            else:
                info['Channel0'] = 0
            if not ('McaCalib' in info):
                if 'MCA a' in info and 'MCA b' in info and 'MCA c' in info:
                    info['McaCalib'] = [float(info['MCA a']),
                                        float(info['MCA b']),
                                        float(info['MCA c'])]
            output.append([info,array])
            #AS self.AppendPage(info,array)
        if len(output) == 1:
            return output[0]
        else:
            return output
        #AS if invalidate: self.Invalidate()

################################################################################
#EXAMPLE CODE:

if __name__ == "__main__":
    import sys,time
    try:
        filename=sys.argv[1]
        key=sys.argv[2]
        fast = int(sys.argv[3])
        obj=EdfFileLayer(fastedf=fast)
        if not obj.SetSource([filename]):
            _logger.error("ERROR: cannot open file %s" % filename)
            sys.exit()
        #obj.LoadSource(key)
    except:
        _logger.error("Usage: EdfFileData.py <filename> <image> <fastflag>")
        sys.exit()
    print(obj.GetSourceInfo())
    for i in range(1):
        #this works obj.LoadSource("%d" % i)
        print("Full")
        e=time.time()
        info,data = obj.LoadSource(key)
        print("elapsed = ",time.time()-e)
        print("selection")
        e=time.time()
        info,data = obj.LoadSource(key,pos=(0,0),size=(90,0))
        print("elapsed = ",time.time()-e)
        print(info)
        #print obj.GetPageInfo("%d" % i)
        #print obj.GetPageInfo(i)
        #print obj.GetPageInfo({'Key':i})
        #print obj.GetPageArray(i)
