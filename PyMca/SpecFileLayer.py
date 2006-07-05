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
    SpecFileLayer.py
    Data derived class to access spec files
"""



################################################################################  
import specfilewrapper as specfile
import Numeric
import string
################################################################################

SOURCE_TYPE = "SpecFile"

# Scan types
# ----------
SF_EMPTY= 0        # empty scan
SF_SCAN        = 1        # non-empty scan
SF_MESH        = 2        # mesh scan
SF_MCA         = 4        # single mca
SF_NMCA        = 8        # multi mca (more than 1 mca per acq)
SF_UMCA        = 16 # mca number does not match pts number


class SpecFileLayer:
    """
    Specializes Data class to access Spec files.
    Interface: Data class interface.
    """
    Error= "SpecFileDataError"

    def __init__(self,refresh_interval=None,info={}):
        """
        See Data.__init__
        """
        info["Class"]="SpecFileData"
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

    def SetSource (self,source_name=None,source_obj=None):
        """
        Sets a new source for data retrieving, an specfile.
        If the file exists, self.Source will be the Specfile
        object associated to this file.
        Parameters:
        source_name: name of the specfile
        """
        if source_name==self.SourceName: return 1
        if source_name is not None:
            if source_obj is not None:
                self.Source= source_obj
            else:
                try:
                    self.Source= specfile.Specfile(source_name)
                except:
                    self.Source= None
        else:
            self.Source= None
        self.SourceInfo= None
        if self.Source is None:
            self.SourceName= None
            return 0
        else:
            self.SourceName= source_name
            return 1
        
    def Refresh(self):
        self.SourceInfo= None
        if self.Source is not None:
                self.Source.update()
        #AS Data.Refresh(self)

    def RefreshPage(source_obj,self,page):
        return 0

    def GetSourceInfo (self, key=None):
        """
        Returns information about the Specfile object created by
        SetSource, to give application possibility to know about
        it before loading.
        Returns a dictionary with the keys "Size" (number of possible
        keys(to this source) and "KeyList" (list of all available keys
        in this source). Each element in "KeyList" is an string in
        the format "x.y" where x is the number of scan and y is
        the order. "x.y" works as the key to retrieve the information
        of this scan.
        There's also the key "NumMca" in the returned dictionary,
        which value is a list of numbers of mcas, for each value of
        "KeyList".
        If key given returns information of a perticular key instead.
        """
        if self.SourceName == None: return None

        if key is None: 
                if self.SourceInfo is None:
                    self.SourceInfo= self.__GetSourceInfo()
                return self.SourceInfo
        else:
            key_type= self.__GetKeyType(key)
            if key_type=="scan": scan_key= key
            elif key_type=="mca": (scan_key, mca_no)=self.__GetMcaPars(key)
            return self.__GetScanInfo(scan_key)
 

    def LoadSource(self,key_list="ALL",append=0,invalidate=1):
        """
        Creates a given number of pages, getting data from the actual 
        source (set by SetSource).
        Parameters:
        * key_list: list of all keys to be read from source. It is a list of strings
              using the following formats:

            "ALL": creates one data page for each scan.
              valid for ScanType==SCAN or MESH or MCA

            "s.o": loads all counter values (s=scan number, o=order)
              - if ScanType==SCAN: in a 2D array (mot*cnts)
              - if ScanType==MESH: in a 3D array (mot1*mot2*cnts)
              - if ScanType==MCA: single MCA in 1D array (0:channels)

            "s.o.n": loads a single MCA in a 1D array (0:channels)
              - if ScanType==NMCA: n is the MCA number from 1 to N
              - if ScanType==SCAN+MCA: n is the scan point number (from 1)
              - if ScanType==MESH+MCA: n is the scan point number (from 1)

            "s.o.p.n": loads a single MCA in a 1D array (0:channels)
              - if ScanType==SCAN+NMCA:
                      p is the point number in the scan
                      n is the MCA device number
              - NOT TRUE: Just follow previous.
                if ScanType==MESH+MCA:
                      p is first motor index
                      n is second motor index

            "s.o.MCA": loads all MCA in an array
              - if ScanType==SCAN+MCA: 2D array (pts*mca)
              - if ScanType==NMCA: 2D array (mca_det*mca)
              - if ScanType==MESH+MCA: 3D array (pts_mot1*pts_mot2*mca)
              - if ScanType==SCAN+NMCA: 3D array (pts_mot1*mca_det*mca)
              - if ScanType==MESH+NMCA:
                      creates N data page, one for each MCA device,
                      with a 3D array (pts_mot1*pts_mot2*mca)

        * append: if non-zero, appends to the end of the page list,
                  Otherwise, initializes the page list
        * invalidate: if non-zero performs an invalidade call after loading
        """
        #AS if append==0: Data.Delete(self)
        if key_list == "ALL": key_list=self.__GetScanList()
        if type(key_list)==type(" "): key_list=[key_list]

        file_info= self.__GetFileInfo()

        output=[]
        for key in key_list:
            key_type= self.__GetKeyType(key)
            if key_type=="scan": output.append(self.__LoadScanData(key,file_info))
            elif key_type=="mca": output.append(self.__LoadMcaData(key,file_info))
        if len(output) == 1:
            return output[0]
        else:
            return output
        #AS if invalidate: self.Invalidate()

    def __LoadScanData(self, scan_key, file_info={}):
        scan_obj= self.Source.select(scan_key)
        scan_info= self.__GetScanInfo(scan_key,scan_obj)
        scan_info["Key"]= scan_key
        scan_info["FileInfo"]= file_info
        scan_type= scan_info["ScanType"]
        scan_data= None

        if scan_type&SF_SCAN:
            try: scan_data= Numeric.transpose(scan_obj.data()).copy()
            except: raise self.Error, "SF_SCAN read failed"
        elif scan_type&SF_MESH:
            try:
                scan_array= scan_obj.data()
                (mot1,mot2,cnts)= self.__GetMeshSize(scan_array)
                scan_data= Numeric.zeros((mot1,mot2,cnts), Numeric.Float)
                for idx in range(mot2):
                    scan_data[:,idx,:]= Numeric.transpose(scan_array[:,idx*mot1:(idx+1)*mot1]).copy()
                scan_data= Numeric.transpose(scan_data).copy()
            except: raise self.Error, "SF_MESH read failed"
        elif scan_type&SF_MCA:
            return self.AppendPage(scan_info, scan_data)
        elif scan_type&SF_NMCA:
            return self.AppendPage(scan_info, scan_data)

        if scan_data is not None:
            return self.AppendPage(scan_info, scan_data)
        else:    raise self.Error, "LoadScanData unknown type"

    def __GetMeshSize(self, scan_array):
        """ Given the scandata array, return the size tuple of the mesh
        """
        mot2_array= scan_array[1]
        mot2_max= mot2_array.shape[0]
        mot1_idx= 1
        while mot1_idx<mot2_max and mot2_array[mot1_idx]==mot2_array[0]: mot1_idx+=1
        mot2_idx= scan_array.shape[1]/mot1_idx
        cnts_idx= scan_array.shape[0]
        return (mot1_idx, mot2_idx, cnts_idx)        

    def __GetScanMotorRange(self, info, obj):
        name= info["LabelNames"][0]
        values= obj.datacol(1)
        length= values.shape[0]
        return (name, values, length)

    def __GetMeshMotorRange(self, info, obj):
        return ()

    def __LoadMcaData(self, key, file_info={}):        
        key_split= key.split(".")
        scan_key= key_split[0]+"."+key_split[1]
        scan_obj= self.Source.select(scan_key)
        scan_info= self.__GetScanInfo(scan_key,scan_obj)
        scan_info["Key"]= key
        scan_info["FileInfo"]= file_info
        scan_type= scan_info["ScanType"]
        scan_data= None
        mca_range= []        # for each dim., (name, length, values or None)

        if key_split[2]=="MCA":
            if scan_type==SF_SCAN+SF_MCA or scan_type==SF_MCA:
                try:
                    mca_length= scan_obj.mca(1).shape[0]
                    scan_data= Numeric.zeros((scan_info["NbMca"], mca_length), Numeric.Float)
                    for idx in range(scan_info["NbMca"]):
                        scan_data[idx]= scan_obj.mca(idx+1)
                    idx= 0
                    if scan_type==SF_SCAN+SF_MCA:
                        mca_range[idx]= self.__GetScanMotorRange(scan_info, scan_obj)
                        idx+=1
                    mca_range[idx]= ("Channels", mca_length, None)
                    scan_info["McaRange"]= mca_range
                    return self.AppendPage(scan_info, scan_data)
                except: 
                    raise self.Error, "SF_SCAN+SF_MCA read failed"
            elif scan_type==SF_NMCA:
                try:
                    mca_length= scan_obj.mca(1).shape[0]
                    mca_det= scan_info["NbMcaDet"]
                    scan_data= Numeric.zeros((mca_det, mca_length), Numeric.Float)
                    for idx in range(mca_det):
                         scan_data[idx]= scan_obj.mca(idx+1)
                    mca_range[0]= ("McaDet", mca_det, None)
                    mca_range[1]= ("Channels", mca_length, None)
                    scan_info["McaRange"]= mca_range
                    return self.AppendPage(scan_info, scan_data)
                except: 
                    raise self.Error, "SF_NMCA read failed"
            elif scan_type==SF_MESH+SF_MCA:
                try:
                    scan_array= scan_obj.data()
                    (mot1,mot2,cnts)= self.__GetMeshSize(scan_array)
                    mca_length= scan_obj.mca(1).shape[0]
                    scan_data= Numeric.zeros((mot1,mot2,mca_length), Numeric.Float)
                    for idx1 in range(mot1):
                        for idx2 in range(mot2):
                            mca_no= 1 + idx1 + idx2*mot1
                            scan_data[idx1,idx2,:]= scan_obj.mca(mca_no)
                        return  self.AppendPage(scan_info, scan_data)
                except: 
                    raise self.Error, "SF_MESH+SF_MCA read failed"
            elif scan_type==SF_SCAN+SF_NMCA:
                try:
                    mca_length= scan_obj.mca(1).shape[0]
                    nbdet= scan_info["NbMcaDet"]
                    nbpts= scan_info["Lines"]
                    scan_data= Numeric.zeros((nbpts, nbdet, mca_length), Numeric.Float)
                    for idx in range(nbpts):
                            for idy in range(nbdet):
                                    scan_data[idx,idy,:]= scan_obj.mca(1+idx*nbdet+idy)
                    mca_range=[0,0,0]
                    mca_range[0]= self.__GetScanMotorRange(scan_info, scan_obj)
                    mca_range[1]= ("McaDet", nbdet, None)
                    mca_range[2]= ("Channels", mca_length, None)
                    scan_info["McaRange"]= mca_range
                    return self.AppendPage(scan_info, scan_data)
                except:
                    raise self.Error, "SF_SCAN+SF_NMCA read failed"
            elif scan_type==SF_MESH+SF_NMCA:
                    raise self.Error, "SF_MESH+SF_NMCA not yet implemented"
                    scan_data= None

        elif len(key_split)==3:
                if scan_type&SF_NMCA or scan_type&SF_MCA:
                #if scan_type==SF_NMCA or \
                #   scan_type==SF_SCAN+SF_MCA or \
                #   scan_type==SF_MESH+SF_MCA:
                        try: 
                                mca_no= int(key_split[2])
                                scan_data= scan_obj.mca(mca_no)
                        except: 
                                raise self.Error, "Single MCA read failed"
                if scan_data is not None:
                        scan_info.update(self.__GetMcaInfo(mca_no, scan_obj, scan_info))
                        return self.AppendPage(scan_info, scan_data)

        elif len(key_split)==4:
                if int(key_split[3]) > scan_info["NbMcaDet"]:
                    raise  self.Error, \
                           "Asked to read Mca %d having % d mca " % \
                           (int(key_split[3]), scan_info["NbMcaDet"])

                if scan_type==SF_SCAN+SF_NMCA:
                    try:
                        mca_no= (int(key_split[2])-1)*scan_info["NbMcaDet"] + int(key_split[3])
                        scan_data= scan_obj.mca(mca_no)
                    except: 
                        raise self.Error, "SF_SCAN+SF_NMCA read failed"
                elif scan_type==SF_MESH+SF_MCA:
                    try:
                        scan_array= scan_obj.data()
                        (mot1,mot2,cnts)= self.__GetMeshSize(scan_array)
                        #mca_no= 1 + int(key_split[2]) + int(key_split[3])*mot1
                        mca_no= (int(key_split[2])-1)*scan_info["NbMcaDet"] + int(key_split[3])
                        scan_data= scan_obj.mca(mca_no)
                    except: 
                        raise self.Error, "SF_MESH+SF_MCA read failed"
                elif scan_type&SF_NMCA or scan_type&SF_MCA:
                    try:
                        mca_no= (int(key_split[2])-1)*scan_info["NbMcaDet"] + int(key_split[3])
                        scan_data= scan_obj.mca(mca_no)
                    except: 
                        raise self.Error, "SF_SCAN+SF_NMCA read failed"
                else:
                    print "Unknown scan!!!!!!!!!!!!!!!!"
                    raise self.Error,"Unknown scan!!!!!!!!!!!!!!!!"
                if scan_data is not None:
                        scan_info.update(self.__GetMcaInfo(mca_no, scan_obj, scan_info))
                        return self.AppendPage(scan_info, scan_data)

    def __GetFileInfo(self):
        file_info={}
        try: file_info["Title"] = self.Source.title()
        except: file_info["Title"] = None
        try: file_info["User"] = self.Source.user()
        except: file_info["User"] = None
        try: file_info["Date"] = self.Source.date()
        except: file_info["Date"] = None
        try: file_info["Epoch"] = self.Source.epoch()
        except: file_info["Epoch"] = None
        try: file_info["ScanNo"] = self.Source.scanno()
        except: file_info["ScanNo"] = None
        try: file_info["List"] = list
        except: file_info["List"] = None
        return file_info

    def __GetSourceInfo(self):
        scanlist=self.__GetScanList()
        source_info={}
        source_info["Size"]=len(scanlist)
        source_info["KeyList"]=scanlist

        num_mca=[]
        num_pts=[]
        commands=[]
        sf_type=[]
        for i in scanlist:
            sel=self.Source.select(i)
            try: n= sel.nbmca()
            except: n= 0
            num_mca.append(n)
            try: n= sel.lines()
            except: n= 0
            num_pts.append(n)
            try: n= sel.command()
            except: n= ""
            commands.append(n)
        source_info["NumMca"]=num_mca
        source_info["NumPts"]=num_pts
        source_info["Commands"]= commands
        source_info["ScanType"]= map(self.__GetScanType, num_pts, num_mca, commands)
        return source_info

    def __GetScanType(self, num_pts, num_mca, command):
        type= SF_EMPTY
        if num_pts>0:
                if command is None:type= SF_SCAN
                elif string.find(command, "mesh")!=-1:
                        type= SF_MESH
                else:   type= SF_SCAN
                if num_mca%num_pts:
                        type+= SF_UMCA
                elif num_mca==num_pts:
                        type+= SF_MCA
                elif num_mca>0:
                        type+= SF_NMCA
        else:
                if num_mca==1:
                        type= SF_MCA
                elif num_mca>1:
                        type= SF_NMCA
        return type

    def __GetScanList(self):
        aux= string.split(self.Source.list(),",")
        newlistcount=[]
        newlist=[]
        for i in aux:
            if string.find(i,":")== -1:  start_index=end_index=int(i)
            else:
                s= string.split(i,":")
                start_index=int(s[0])
                end_index=int(s[1])
            for j in range(start_index,end_index+1):
                newlist.append(j)
                newlistcount.append(newlist.count(j))
        for i in range(len(newlist)):
            newlist[i]="%d.%d" % (newlist[i],newlistcount[i])
        return newlist
 

    def __GetKeyType (self,key):
        count= string.count(key, '.')
        if (count==1): return "scan"
        elif (count==2) or (count==3): return "mca"
        else: raise "SpecFileData: Invalid key"


    def __GetScanInfo(self, scankey, scandata=None):
        if scandata is None: scandata= self.Source.select(scankey)

        info={}
        info["SourceType"]= SOURCE_TYPE
        info["SourceName"]=self.SourceName
        info["Key"]=scankey
        info["Source"]=self.Source

        try: info["Number"] = scandata.number()
        except: info["Number"] = None
        try: info["Order"] = scandata.order()
        except: info["Order"] = None
        try: info["Cols"] = scandata.cols()
        except: info["Cols"] = 0
        try: info["Lines"] = scandata.lines()
        except: info["Lines"] = 0
        try: info["Date"] = scandata.date()
        except: info["Date"] = None
        try: info["MotorNames"] = self.Source.allmotors()
        except: info["MotorNames"] = None
        try: info["MotorValues"] = scandata.allmotorpos()
        except: info["MotorValues"] = None
        try: info["LabelNames"] = scandata.alllabels()
        except: info["LabelNames"] = None
        try: info["Command"] = scandata.command()
        except: info["Command"] = None
        try: info["Header"] = scandata.header("")
        except: info["Header"] = None
        try: info["NbMca"] = scandata.nbmca()
        except: info["NbMca"] = 0
        try: info["Hkl"] =  scandata.hkl()
        except: info["Hkl"] =  None
        if info["NbMca"]:
            if info["Lines"]>0 and info["NbMca"]%info["Lines"]==0:
                info["NbMcaDet"]= info["NbMca"]/info["Lines"]
            else:
                info["NbMcaDet"]= info["NbMca"]
        info["ScanType"]= self.__GetScanType(info["Lines"], info["NbMca"], info["Command"])
        return info


    def __GetMcaInfo(self, mcano, scandata, info={}):
        mcainfo= {}
        if info.has_key("NbMcaDet"):
            det= info["NbMcaDet"]
            if info["Lines"]>0:
                mcainfo["McaPoint"]= int(mcano/info["NbMcaDet"])+(mcano%info["NbMcaDet"]>0)
                mcainfo["McaDet"]= mcano-((mcainfo["McaPoint"]-1)*info["NbMcaDet"])
                try: mcainfo["LabelValues"]= scandata.dataline(mcainfo["McaPoint"])
                except: mcainfo["LabelValues"]= None
            else:
                mcainfo["McaPoint"]= 0
                mcainfo["McaDet"]= mcano
                mcainfo["LabelValues"]= None
        calib= scandata.header("@CALIB")
        mcainfo["McaCalib"]=[0.0,1.0,0.0]
        if len(calib):
                ctxt= calib[0].split()
                if len(ctxt)==4:
                        try:
                                cval= [ float(ctxt[1]), float(ctxt[2]), float(ctxt[3]) ]
                                mcainfo["McaCalib"]= cval
                        except: 
                            mcainfo["McaCalib"]=[0.0,1.0,0.0]
        ctime= scandata.header("@CTIME")
        if len(ctime):
                ctxt= ctime[0].split()
                if len(ctxt)==4:
                        try:
                                mcainfo["McaPresetTime"]= float(ctxt[1])
                                mcainfo["McaLiveTime"]= float(ctxt[2])
                                mcainfo["McaRealTime"]= float(ctxt[3])
                        except: pass
                        
        chann = scandata.header("@CHANN")
        if len(chann):
            ctxt= chann[0].split()
            if len(ctxt)==5:
                mcainfo['Channel0'] = float(ctxt[2])
            else:
                mcainfo['Channel0'] = 0.0
        else:
            mcainfo['Channel0'] = 0.0                
        return mcainfo


    def __GetMcaPars(self,key):
        nums= string.split(key,'.')
        size = len(nums)
        sel_key = nums[0] + "." + nums[1]
        if size==3:
            mca_no=int(nums[2])
        elif size==4:
            sel=self.Source.select(sel_key)
            try: lines = sel.lines()
            except: lines=0
            if nums[3]==0: mca_no=int(nums[2])
            else:          mca_no=((int(nums[3])-1)*lines)+int(nums[2])
        else: raise "SpecFileData: Invalid key"
        return (sel_key,mca_no)



################################################################################
#EXEMPLE CODE:
 
if __name__ == "__main__":
    import sys,time

    if len(sys.argv) not in [2,3]:
        print "Usage: %s <filename> [<key_to_load>]"
        sys.exit()

    filename= sys.argv[1]
    sf= SpecFileLayer()
    if not sf.SetSource(filename):
        print "ERROR: cannot open file", filename
        sys.exit()
    if len(sys.argv)==2:
        info= sf.GetSourceInfo()
        print "Filename        :", sf.SourceName
        print "Number of scans :", info["Size"]

        print "S# - command - pts - mca - type"
        for (s,c,p,m,t) in zip(info["KeyList"],info["Commands"],info["NumPts"],info["NumMca"],info["ScanType"]):
                print s,"-",c,"-",p,"-",m,"-",t
        print "KeyList = ",info["KeyList"]
        #print info['Channel0']

    if len(sys.argv)==3:
        info, data = sf.LoadSource(sys.argv[2])

        print "Filename   :", sf.SourceName
        print "Loaded key :", info["Key"]
        print "Header     :"
        for i,v in info.items():
                print "-", i, ":", v
        print "Data Shape :", data.shape



 
