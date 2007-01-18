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
__revision__ = "$Revision: 1.30 $"
import ClassMcaTheory
import SpecFileLayer
import EdfFileLayer
import EdfFile
import Numeric
import os
import sys
import ConfigDict
import ConcentrationsTool


class McaAdvancedFitBatch:
    def __init__(self,initdict,filelist=None,outputdir=None,
                    roifit=None,roiwidth=None,
                    overwrite=1, filestep=1, mcastep=1,
                    concentrations=0, fitfiles=1, fitimages=1):
        #for the time being the concentrations are bound to the .fit files
        #that is not necessary, but it will be correctly implemented in
        #future releases
        self.fitFiles = fitfiles
        self._concentrations = concentrations
        if type(initdict) == type([]):
            self.mcafit = ClassMcaTheory.McaTheory(initdict[0])
            self.__configList = initdict
        else:
            self.__configList = None
            self.mcafit = ClassMcaTheory.McaTheory(initdict)
        if self._concentrations:
            self._tool = ConcentrationsTool.ConcentrationsTool()
            self._toolConversion = ConcentrationsTool.ConcentrationsConversion()
        self.setFileList(filelist)
        self.setOutputDir(outputdir)
        if fitimages:
            self.fitImages=  1
            self.__ncols  =  None
        else:
            self.fitImages = False
            self.__ncols = None
        self.fileStep = filestep
        self.mcaStep  = mcastep 
        self.useExistingFiles = not overwrite
        self.savedImages=[]
        if roifit   is None:roifit   = False
        if roiwidth is None:roiwidth = 100.
        self.pleaseBreak = 0
        self.roiFit   = roifit
        self.roiWidth = roiwidth

        
    def setFileList(self,filelist=None):
        self._rootname = ""
        if filelist is None:filelist = []
        if type(filelist) == type(" "):filelist = [filelist]
        self._filelist = filelist
        self._rootname = self.getRootName(filelist)

    def getRootName(self,filelist=None):
        if filelist is None:filelist = self._filelist
        first = os.path.basename(filelist[ 0])
        last  = os.path.basename(filelist[-1])
        if first == last:return os.path.splitext(first)[0]
        name1,ext1 = os.path.splitext(first)
        name2,ext2 = os.path.splitext(last )
        i0=0
        for i in range(len(name1)):
            if name1[i] == name2[i]:
                pass
            else:
                break
        i0 = i
        for i in range(i0,len(name1)):
            if name1[i] != name2[i]:
                pass
            else:
                break
        i1 = i
        if i1 > 0:
            delta=1
            while (i1-delta):
                if (last[(i1-delta)] in ['0', '1', '2',
                                        '3', '4', '5',
                                        '6', '7', '8',
                                        '9']):
                    delta = delta + 1
                else:
                    if delta > 1: delta = delta -1
                    break
            rootname = name1[0:]+"_to_"+last[(i1-delta):]
        else:
            rootname = name1[0:]+"_to_"+last[0:]            
        return rootname

    def setOutputDir(self,outputdir=None):
        if outputdir is None:outputdir=os.getcwd()
        self._outputdir = outputdir
        
    def processList(self):
        self.counter =  0
        self.__row   = -1
        for i in range(0,len(self._filelist),self.fileStep):
            if not self.roiFit:
                if self.__configList is not None:
                    if i != 0:
                        self.mcafit = ClassMcaTheory.McaTheory(self.__configList[i])
            self.mcafit.enableOptimizedLinearFit()
            inputfile   = self._filelist[i]
            self.__row += 1
            self.onNewFile(inputfile, self._filelist)
            self.file = self.getFileHandle(inputfile)
            if self.pleaseBreak: break
            self.__processOneFile()
        if self.counter:
            if not self.roiFit: 
                if self.fitFiles:
                    self.listfile.write(']\n')
                    self.listfile.close()
            if self.__ncols is not None:
                if self.__ncols:self.saveImage()
        self.onEnd()

    def getFileHandle(self,inputfile):
        try:
            ffile = self.__tryEdf(inputfile)
            if (ffile is None):
                del ffile
                ffile   = SpecFileLayer.SpecFileLayer()
                ffile.SetSource(inputfile)
            return ffile
        except:
            raise "IOerror","I do not know what to do with file %s" % inputfile        
    
    
    def onNewFile(self,ffile, filelist):
        self.__log(ffile)

    def onImage(self,image,imagelist):
        pass
        
    def onMca(self,mca,nmca, filename=None, key=None, info=None):
        pass


    def onEnd(self):
        pass
            
    def __log(self,text):
        print text
            
    def __tryEdf(self,inputfile):
        try:
            ffile   = EdfFileLayer.EdfFileLayer(fastedf=0)
            ffile.SetSource(inputfile)
            fileinfo = ffile.GetSourceInfo()
            if fileinfo['KeyList'] == []:ffile=None
            return ffile
        except:
            return None

    def __processOneFile(self):
        ffile=self.file
        fileinfo = ffile.GetSourceInfo()
        nimages = nscans = len(fileinfo['KeyList'])
        if 1:
            i = 0
            for scankey in  fileinfo['KeyList']:
                if self.pleaseBreak: break
                self.onImage(scankey, fileinfo['KeyList'])
                if 0:
                    scan,rc   = string.split(scankey,".")
                    info,data  = ffile.LoadSource({'Key':int(image)-1})
                else:
                    scan,order = scankey.split(".")
                    info,data  = ffile.LoadSource(scankey)
                if info['SourceType'] == "EdfFile":
                    nrows = int(info['Dim_1'])
                    ncols = int(info['Dim_2'])
                    numberofmca  = min(nrows,ncols)
                    self.__ncols = len(range(0,numberofmca,self.mcaStep))
                    self.__col  = -1
                    for mca in range(0,numberofmca,self.mcaStep):
                        if self.pleaseBreak: break
                        self.__col += 1
                        if int(nrows) > int(ncols):
                            row=mca
                            col=0
                            mcadata = data[mca,:]
                        else:
                            col=mca
                            row=0
                            mcadata = data[:,mca]
                        if info.has_key('MCA start ch'):
                            xmin = float(info['MCA start ch'])
                        else:
                            xmin = 0.0
                        #key = "%s.%s.%02d.%02d" % (scan,order,row,col)
                        key = "%s.%s.%04d" % (scan,order,mca)
                        if 0:
                            #slow
                            y0  = Numeric.array(mcadata.tolist())
                        else:
                            #fast
                            y0  = Numeric.array(mcadata)
                        x = Numeric.arange(len(y0))*1.0 + xmin
                        filename = os.path.basename(info['SourceName'])
                        infoDict = {}
                        infoDict['SourceName'] = info['SourceName']
                        infoDict['Key']        = key
                        self.__processOneMca(x,y0,filename,key,info=infoDict)
                        self.onMca(mca, numberofmca, filename=filename,
                                                    key=key,
                                                    info=infoDict)
                else:
                    if info['NbMca'] > 0:
                        self.fitImages = True
                        self.__ncols = info['NbMca'] * 1
                        self.__col   = -1
                        scan_key = "%s.%s" % (scan,order)
                        scan_obj= ffile.Source.select(scan_key)
                        #import time                        
                        for i in range(info['NbMca']):
                            #e0 = time.time()
                            if self.pleaseBreak: break
                            self.__col += 1
                            point = int(i/info['NbMcaDet']) + 1
                            mca   = (i % info['NbMcaDet'])  + 1
                            key = "%s.%s.%05d.%d" % (scan,order,point,mca)
                            #get rid of slow info reading methods
                            #mcainfo,mcadata = ffile.LoadSource(key)
                            mcadata = scan_obj.mca(i+1)
                            y0  = Numeric.array(mcadata)
                            x = Numeric.arange(len(y0))*1.0
                            filename = os.path.basename(info['SourceName'])

                            infoDict = {}
                            infoDict['SourceName'] = info['SourceName']
                            infoDict['Key']        = key
                            self.__processOneMca(x,y0,filename,key,info=infoDict)
                            self.onMca(i, info['NbMca'],filename=filename,
                                                    key=key,
                                                    info=infoDict)
                            #print "remaining = ",(time.time()-e0) * (info['NbMca'] - i)

    def __getFitFile(self, filename, key):
           fitdir = os.path.join(self._outputdir,"FIT")
           fitdir = os.path.join(fitdir,filename+"_FITDIR")
           outfile = filename +"_"+key+".fit" 
           outfile = os.path.join(fitdir,  outfile)           
           return outfile

                
    def __processOneMca(self,x,y,filename,key,info=None):
        self._concentrationsAsAscii = ""
        if not self.roiFit:
            result = None
            concentrationsdone = 0
            concentrations = None
            outfile=os.path.join(self._outputdir,filename)
            fitfile = self.__getFitFile(filename,key)
            self._concentrationsFile = os.path.join(self._outputdir,
                                    self._rootname+"_concentrations.txt")
            #                        self._rootname+"_concentrationsNEW.txt")
            if self.counter == 0:
                if os.path.exists(self._concentrationsFile):
                    try:
                        os.remove(self._concentrationsFile)
                    except:
                        print "I could not delete existing concentrations file %s", self._concentrationsFile                        
            #print "self._concentrationsFile", self._concentrationsFile
            if self.useExistingFiles and os.path.exists(fitfile):
                useExistingResult = 1
                try:
                    dict = ConfigDict.ConfigDict()
                    dict.read(fitfile)
                    result = dict['result']
                    if dict.has_key('concentrations'):
                        concentrationsdone = 1
                except:
                    print "Error trying to use result file %s" % fitfile
                    print "Please, consider deleting it."
                    print sys.exc_info()
                    return
            else:
                useExistingResult = 0
                try:
                    #I make sure I take the fit limits configuration
                    self.mcafit.config['fit']['use_limit'] = 1
                    self.mcafit.setdata(x,y)
                except:
                    print "Error entering data of file with output = ",filename
                    return
                try:
                    self.mcafit.estimate()
                    if self.fitFiles:
                        fitresult, result = self.mcafit.startfit(digest=1)
                    elif self._concentrations and (self.mcafit._fluoRates is None):
                        fitresult, result = self.mcafit.startfit(digest=1)
                    elif self._concentrations:
                        fitresult = self.mcafit.startfit(digest=0)
                        try:
                            fitresult0 = {}
                            fitresult0['fitresult'] = fitresult
                            fitresult0['result'] = self.mcafit.imagingDigestResult()
                            fitresult0['result']['config'] = self.mcafit.config
                            tconf = self._tool.configure()
                            concentrations = self._tool.processFitResult(config=tconf,
                                            fitresult=fitresult0,
                                            elementsfrommatrix=False,
                                            fluorates = self.mcafit._fluoRates)
                        except:
                            print "error in concentrations"
                            print sys.exc_info()[0:-1]
                        concentrationsdone = True
                    else:
                        #just images
                        fitresult = self.mcafit.startfit(digest=0)
                except:
                    print "Error fitting file with output = ",filename
                    return
            if self._concentrations:
                if concentrationsdone == 0:
                    if not result.has_key('concentrations'):
                        if useExistingResult:
                            fitresult0={}
                            fitresult0['result'] = result
                            conf = result['config']
                        else:
                            fitresult0={}
                            if result is None:
                                result = self.mcafit.digestresult()
                            fitresult0['result']    = result
                            fitresult0['fitresult'] = fitresult
                            conf = self.mcafit.configure()
                        tconf = self._tool.configure()
                        if conf.has_key('concentrations'):
                            tconf.update(conf['concentrations'])
                        else:
                            pass
                            #print "Concentrations not calculated"
                            #print "Is your fit configuration file correct?"
                            #return                      
                        try:
                            concentrations = self._tool.processFitResult(config=tconf,
                                            fitresult=fitresult0,
                                            elementsfrommatrix=False)
                        except:
                            print "error in concentrations"
                            print sys.exc_info()[0:-1]
                            #return
                self._concentrationsAsAscii=self._toolConversion.getConcentrationsAsAscii(concentrations)
                if len(self._concentrationsAsAscii) > 1:
                    text  = ""
                    text += "SOURCE: "+ filename +"\n"
                    text += "KEY: "+key+"\n"
                    text += self._concentrationsAsAscii + "\n"
                    f=open(self._concentrationsFile,"a")
                    f.write(text)
                    f.close()

            #output options
            # .FIT files
            if self.fitFiles:
                fitdir = os.path.join(self._outputdir,"FIT")
                if not os.path.exists(fitdir):
                    try:
                        os.mkdir(fitdir)
                    except:
                        print "I could not create directory %s" % fitdir
                        return
                fitdir = os.path.join(fitdir,filename+"_FITDIR")
                if not os.path.exists(fitdir):
                    try:
                        os.mkdir(fitdir)
                    except:
                        print "I could not create directory %s" % fitdir
                        return
                if not os.path.isdir(fitdir):
                    print "%s does not seem to be a valid directory" % fitdir
                else:
                    outfile = filename +"_"+key+".fit" 
                    outfile = os.path.join(fitdir,  outfile)
                if not useExistingResult:
                    result = self.mcafit.digestresult(outfile=outfile,
                                                      info=info)
                if concentrations is not None:
                    try:
                        f=ConfigDict.ConfigDict()
                        f.read(outfile)
                        f['concentrations'] = concentrations
                        try:
                            os.remove(outfile)
                        except:
                            print "error deleting fit file"
                        f.write(outfile)
                    except:
                        print "Error writing concentrations to fit file"
                        print sys.exc_info()

                #python like output list
                if not self.counter:
                    name = os.path.splitext(self._rootname)[0]+"_fitfilelist.py"
                    name = os.path.join(self._outputdir,name)
                    try:
                        os.remove(name)
                    except:
                        pass
                    self.listfile=open(name,"w+")
                    self.listfile.write("fitfilelist = [")
                    self.listfile.write('\n'+outfile)
                else: 
                    self.listfile.write(',\n'+outfile)
            else:
                if not useExistingResult:
                    if 0:
                        #this is very slow and not needed just for imaging
                        if result is None:result = self.mcafit.digestresult()
                    else:
                        if result is None:result = self.mcafit.imagingDigestResult()

            #IMAGES
            if self.fitImages:
                #this only works with EDF
                if self.__ncols is not None:
                    if not self.counter:
                        imgdir = os.path.join(self._outputdir,"IMAGES")
                        if not os.path.exists(imgdir):
                            try:
                                os.mkdir(imgdir)
                            except:
                                print "I could not create directory %s" % imgdir
                                return
                        elif not os.path.isdir(imgdir):
                            print "%s does not seem to be a valid directory"
                        self.imgDir = imgdir
                        self.__peaks  = []
                        self.__images = {}
                        self.__sigmas = {}
                        self.__nrows   = len(range(0,len(self._filelist),self.fileStep))
                        for group in result['groups']:
                            self.__peaks.append(group)
                            self.__images[group]=Numeric.zeros((self.__nrows,self.__ncols),Numeric.Float)
                            self.__sigmas[group]=Numeric.zeros((self.__nrows,self.__ncols),Numeric.Float)
                        self.__images['chisq']  = Numeric.zeros((self.__nrows,self.__ncols),Numeric.Float) - 1.
                for peak in self.__peaks:
                    try:
                        self.__images[peak][self.__row, self.__col] = result[peak]['fitarea']
                        self.__sigmas[peak][self.__row, self.__col] = result[peak]['sigmaarea']
                    except:
                        pass
                try:
                    self.__images['chisq'][self.__row, self.__col] = result['chisq']
                except:
                    print "Error on chisq row %d col %d\n" % (self.__row, self.__col)
                    pass

        else:
                dict=self.mcafit.roifit(x,y,width=self.roiWidth)
                #this only works with EDF
                if self.__ncols is not None:
                    if not self.counter:
                        imgdir = os.path.join(self._outputdir,"IMAGES")
                        if not os.path.exists(imgdir):
                            try:
                                os.mkdir(imgdir)
                            except:
                                print "I could not create directory %s" % imgdir
                                return
                        elif not os.path.isdir(imgdir):
                            print "%s does not seem to be a valid directory"
                        self.imgDir = imgdir
                        self.__ROIpeaks  = []
                        self._ROIimages = {}
                        self.__nrows   = len(self._filelist)
                        for group in dict.keys():
                            self.__ROIpeaks.append(group)
                            self._ROIimages[group]={}
                            for roi in dict[group].keys():
                                self._ROIimages[group][roi]=Numeric.zeros((self.__nrows,
                                                                            self.__ncols),Numeric.Float)
                                
                if not hasattr(self, "_ROIimages"):
                    print "ROI fitting only supported on EDF"
                for group in self.__ROIpeaks:
                    for roi in self._ROIimages[group].keys():
                        try:
                            self._ROIimages[group][roi][self.__row, self.__col] = dict[group][roi]
                        except:
                            print "error on (row,col) = ",self.__row, self.__col
                            pass

        #update counter
        self.counter += 1

            
    def saveImage(self,ffile=None):
        self.savedImages=[]
        if ffile is None:
            ffile = os.path.splitext(self._rootname)[0]
            ffile = os.path.join(self.imgDir,ffile)
        if not self.roiFit:
            if (self.fileStep > 1) or (self.mcaStep > 1):
                trailing = "_filestep_%02d_mcastep_%02d" % ( self.fileStep,
                                                             self.mcaStep )
            else:
                trailing = ""
            #speclabel = "#L row  column"
            speclabel = "row  column"
            for peak in (self.__peaks+['chisq']):
                if peak != 'chisq':
                    a,b = peak.split()
                    speclabel +="  %s" % (a+"-"+b)
                    speclabel +="  s(%s)" % (a+"-"+b)
                    edfname = ffile +"_"+a+"_"+b+trailing+".edf"
                else:
                    speclabel +="  %s" % (peak)
                    edfname = ffile +"_"+peak+trailing+".edf"
                dirname = os.path.dirname(edfname)
                if not os.path.exists(dirname):
                    try:
                        os.mkdir(dirname)
                    except:
                        print "I could not create directory % s" % dirname
                edfout   = EdfFile.EdfFile(edfname)
                edfout.WriteImage ({'Title':peak} , self.__images[peak], Append=0)
                self.savedImages.append(edfname)
            #save specfile format
            specname = ffile+trailing+".dat"
            if os.path.exists(specname):
                try:
                    os.remove(specname)
                except:
                    pass
            specfile=open(specname,'w+')
            #specfile.write('\n')
            #specfile.write('#S 1  %s\n' % (file+trailing))
            #specfile.write('#N %d\n' % (len(self.__peaks)+2))
            specfile.write('%s\n' % speclabel)
            specline=""
            for row in range(self.__nrows):
                for col in range(self.__ncols):
                    specline += "%d" % row
                    specline += "  %d" % col
                    for peak in self.__peaks:
                        #write area
                        specline +="  %g" % self.__images[peak][row][col]
                        #write sigma area
                        specline +="  %g" % self.__sigmas[peak][row][col]
                    #write global chisq
                    specline +="  %g" % self.__images['chisq'][row][col]
                    specline += "\n"
                    specfile.write("%s" % specline)
                    specline =""
            specfile.write("\n") 
            specfile.close()
        else:
            for group in self.__ROIpeaks:
                i = 0
                grouptext = group.replace(" ","_")
                for roi in self._ROIimages[group].keys():
                    #roitext = roi.replace(" ","-")
                    if (self.fileStep > 1) or (self.mcaStep > 1):
                        edfname = ffile+"_"+grouptext+("_%04deVROI_filestep_%02d_mcastep_%02d.edf" % (self.roiWidth,
                                                                    self.fileStep, self.mcaStep ))
                    else: 
                        edfname = ffile+"_"+grouptext+("_%04deVROI.edf" % self.roiWidth)
                    dirname = os.path.dirname(edfname)
                    if not os.path.exists(dirname):
                        try:
                            os.mkdir(dirname)
                        except:
                            print "I could not create directory % s" % dirname
                    edfout  = EdfFile.EdfFile(edfname)
                    edfout.WriteImage ({'Title':group+" "+roi} , self._ROIimages[group][roi],
                                         Append=i)
                    if i==0:
                        self.savedImages.append(edfname)
                        i=1
                    

if __name__ == "__main__":
    import getopt
    options     = 'f'
    longoptions = ['cfg=','pkm=','outdir=','roifit=','roi=','roiwidth=']
    filelist = None
    outdir   = None
    cfg      = None
    roifit   = 0
    roiwidth = 250.
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)
    for opt,arg in opts:
        if opt in ('--pkm','--cfg'):
            cfg = arg
        elif opt in ('--outdir'):
            outdir = arg
        elif opt in ('--roi','--roifit'):
            roifit   = int(arg)
        elif opt in ('--roiwidth'):
            roiwidth = float(arg)
    filelist=args
    if len(filelist) == 0:
        print "No input files, run GUI"
        sys.exit(0)
    
    b = McaAdvancedFitBatch(cfg,filelist,outdir,roifit,roiwidth)
    b.processList()
