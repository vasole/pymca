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
import sys
import os
import time
import numpy
from . import ClassMcaTheory
from PyMca5.PyMcaCore import SpecFileLayer
from PyMca5.PyMcaCore import EdfFileLayer
from PyMca5.PyMcaIO import EdfFile
from PyMca5.PyMcaIO import LuciaMap
from PyMca5.PyMcaIO import AifiraMap
from PyMca5.PyMcaIO import EDFStack
from PyMca5.PyMcaIO import LispixMap
from PyMca5.PyMcaIO import NumpyStack
try:
    import h5py
    from PyMca5.PyMcaIO import HDF5Stack1D
    HDF5SUPPORT = True
except ImportError:
    HDF5SUPPORT = False
from PyMca5.PyMcaIO import ConfigDict
from . import ConcentrationsTool
from .XRFBatchFitOutput import OutputBuffer


class McaAdvancedFitBatch(object):

    def __init__(self, initdict, filelist=None, outputdir=None,
                 roifit=False, roiwidth=100,
                 overwrite=1, filestep=1, mcastep=1,
                 fitfiles=1, fitimages=1,
                 concentrations=0, fitconcfile=1,
                 filebeginoffset=0, fileendoffset=0,
                 mcaoffset=0, chunk=None,
                 selection=None, lock=None, nosave=None,
                 quiet=False, outbuffer=None):
        #for the time being the concentrations are bound to the .fit files
        #that is not necessary, but it will be correctly implemented in
        #future releases
        self._lock = lock

        self.setFileList(filelist)
        self.__ncols = None
        self.fileStep = filestep
        self.mcaStep = mcastep
        self.savedImages = []
        self.pleaseBreak = 0
        self.roiFit = roifit
        self.roiWidth = roiwidth
        self.fileBeginOffset = filebeginoffset
        self.fileEndOffset = fileendoffset
        self.mcaOffset = mcaoffset
        self.chunk = chunk
        self.selection = selection
        self.quiet = quiet
        self.fitFiles = fitfiles
        self.fitConcFile = fitconcfile
        self._concentrations = concentrations

        if isinstance(initdict, list):
            self.mcafit = ClassMcaTheory.McaTheory(initdict[mcaoffset])
            self.__configList = initdict
            self.__currentConfig = mcaoffset
        else:
            self.__configList = [initdict]
            self.__currentConfig = 0
            self.mcafit = ClassMcaTheory.McaTheory(initdict)
        self.__concentrationsKeys = []
        if self._concentrations:
            self._tool = ConcentrationsTool.ConcentrationsTool()
            self._toolConversion = ConcentrationsTool.ConcentrationsConversion()

        self.overwrite = overwrite
        self._nosave = bool(nosave)  # TODO: to be removed
        self.outputdir = outputdir
        self.outbuffer = outbuffer
        if fitimages:
            self._initOutputBuffer()

    @property
    def useExistingFiles(self):
        return not self.overwrite

    def _initOutputBuffer(self):
        if self.outbuffer is None:
            self.outbuffer = OutputBuffer(outputDir=self.outputdir,
                                          outputRoot=self._rootname,
                                          fileEntry=self._rootname,
                                          overwrite=self.overwrite,
                                          h5=True, edf=True, dat=True,
                                          suffix=self._outputSuffix())
        self.outbuffer['configuration'] = self.mcafit.getConfiguration()

    def _outputSuffix(self):
        suffix = ""
        if self.roiFit:
            suffix = "_%04deVROI" % self.roiWidth 
        if (self.fileStep > 1) or (self.mcaStep > 1):
            suffix += "_filestep_%02d_mcastep_%02d" %\
                        (self.fileStep, self.mcaStep)
        if self.chunk is not None:
            suffix += "_%06d_partial" % self.chunk
        return suffix

    def setFileList(self, filelist=None):
        self._rootname = ""
        if filelist is None:
            filelist = []
        if type(filelist) not in [type([]), type((2,))]:
            filelist = [filelist]
        self._filelist = filelist
        if len(filelist):
            if type(filelist[0]) is not numpy.ndarray:
                self._rootname = self.getRootName(filelist)

    def getRootName(self, filelist=None):
        if filelist is None:
            filelist = self._filelist
        first = os.path.basename(filelist[ 0])
        last = os.path.basename(filelist[-1])
        if first == last:
            return os.path.splitext(first)[0]
        name1, ext1 = os.path.splitext(first)
        name2, ext2 = os.path.splitext(last)
        i0 = 0
        for i in range(len(name1)):
            if i >= len(name2):
                break
            elif name1[i] == name2[i]:
                pass
            else:
                break
        i0 = i
        for i in range(i0, len(name1)):
            if i >= len(name2):
                break
            elif name1[i] != name2[i]:
                pass
            else:
                break
        i1 = i
        if i1 > 0:
            delta = 1
            while (i1-delta):
                if (last[i1-delta] in ['0', '1', '2',
                                        '3', '4', '5',
                                        '6', '7', '8',
                                        '9']):
                    delta = delta + 1
                else:
                    if delta > 1:
                        delta = delta - 1
                    break
            rootname = name1[0:]+"_to_"+name2[(i1-delta):]
        else:
            rootname = name1[0:]+"_to_"+name2[0:]
        return rootname

    @property
    def outputdir(self):
        return self._outputdir
    
    @outputdir.setter
    def outputdir(self, value):
        if value is None:
            value = os.getcwd()
        self._outputdir = value
        self._obsoleteUpdateImgDir()

    def _obsoleteUpdateImgDir(self):
        if self._nosave:
            self.imgDir = None
        else:
            imgdir = self.os_path_join(self.outputdir, "IMAGES")
            if not os.path.exists(imgdir):
                try:
                    os.mkdir(imgdir)
                except:
                    print("I could not create directory %s" %\
                          imgdir)
                    return
            elif not os.path.isdir(imgdir):
                print("%s does not seem to be a valid directory" %\
                      imgdir)
            self.imgDir = imgdir

    def processList(self):
        self.counter = 0
        self.__row = self.fileBeginOffset - 1
        self.__stack = None
        self.listfile = None
        with self.outbuffer.saveContext():
            start = 0+self.fileBeginOffset
            stop = len(self._filelist)-start
            for i in range(start, stop, self.fileStep):
                if not self.roiFit:
                    if len(self.__configList) > 1:
                        if i != 0:
                            self.mcafit = ClassMcaTheory.McaTheory(self.__configList[i])
                            self.__currentConfig = i
                            # TODO: outbuffer does not support multiple configurations
                self.mcafit.enableOptimizedLinearFit()

                inputfile = self._filelist[i]
                self.__row += 1  #should be plus fileStep?
                self.onNewFile(inputfile, self._filelist)
                self.file = self.getFileHandle(inputfile)
                if self.pleaseBreak:
                    break
                if self.__stack is None:
                    self.__stack = False
                    if hasattr(self.file, "info"):
                        if "SourceType" in self.file.info:
                            if self.file.info["SourceType"] in\
                            ["EdfFileStack", "HDF5Stack1D"]:
                                self.__stack = True
                if self.__stack:
                    self.__processStack()
                    if self._HDF5:
                        # The complete stack has been analyzed
                        break
                else:
                    self.__processOneFile()

            if self.counter:
                # Finish list of FIT files
                if not self.roiFit and self.fitFiles and \
                   self.listfile is not None:
                        self.listfile.write(']\n')
                        self.listfile.close()
                # Save results as .edf and .dat
                if self.__ncols and (not self._nosave):
                    self._obsoleteSaveImage()  # TODO: remove
        self.onEnd()

    def getFileHandle(self,inputfile):
        try:
            self._HDF5 = False
            if type(inputfile) == numpy.ndarray:
                try:
                    a = NumpyStack.NumpyStack(inputfile)
                    return a
                except Exception as e:
#                     print e
                    raise
        
            if HDF5SUPPORT:
                if h5py.is_hdf5(inputfile):
                    self._HDF5 = True
                    try:
                        # if (len(self._filelist) == 1) && (self.mcaStep > 1)
                        # it should attempt to avoid loading  many times
                        # the stack into memory in case of multiple processes
                        return HDF5Stack1D.HDF5Stack1D(self._filelist,
                                                       self.selection)
                    except:
                        raise
            
            ffile = self.__tryEdf(inputfile)
            if ffile is None:
                ffile = self.__tryLucia(inputfile)
            if ffile is None:
                if inputfile[-3:] == "DAT":
                    ffile = self.__tryAifira(inputfile)
            if ffile is None:
                if LispixMap.isLispixMapFile(inputfile):
                    ffile = LispixMap.LispixMap(inputfile, native=False)
            if (ffile is None):
                del ffile
                ffile   = SpecFileLayer.SpecFileLayer()
                ffile.SetSource(inputfile)
            return ffile
        except:
            raise IOError("I do not know what to do with file %s" % inputfile)


    def onNewFile(self,ffile, filelist):
        if not self.quiet:
            self.__log(ffile)

    def onImage(self,image,imagelist):
        pass

    def onMca(self,mca,nmca, filename=None, key=None, info=None):
        pass

    def onEnd(self):
        pass

    def __log(self,text):
        print(text)

    def __tryEdf(self,inputfile):
        try:
            ffile   = EdfFileLayer.EdfFileLayer(fastedf=0)
            ffile.SetSource(inputfile)
            fileinfo = ffile.GetSourceInfo()
            if fileinfo['KeyList'] == []:
                ffile=None
            elif len(self._filelist) == 1:
                #Is it a Diamond stack?
                if len(fileinfo['KeyList']) > 1:
                    info, data = ffile.LoadSource(fileinfo['KeyList'][0])
                    shape = data.shape
                    if len(shape) == 2:
                        if min(shape) == 1:
                            #It is a Diamond Stack
                            ffile = EDFStack.EDFStack(inputfile)
            return ffile
        except:
            return None

    def __tryLucia(self, inputfile):
        f = open(inputfile)
        line = f.readline()
        f.close()
        ffile = None
        if line.startswith('#\tDate:'):
            ffile = LuciaMap.LuciaMap(inputfile)
        return ffile

    def __tryAifira(self, inputfile):
        if sys.platform == "win32":
            f = open(inputfile,"rb")
        else:
            f = open(inputfile,"r")
        line = f.read(3)
        f.close()
        if '#' in line:
            #specfile
            return None
        ffile = None
        try:
            ffile = AifiraMap.AifiraMap(inputfile)
        except:
            ffile = None
        return ffile

    def __processStack(self):
        stack = self.file
        info = stack.info
        data = stack.data
        xStack = None
        if hasattr(stack, "x"):
            if stack.x not in [None, []]:
                if type(stack.x) == type([]):
                    xStack = stack.x[0]
                else:
                    print("THIS SHOULD NOT BE USED")
                    xStack = stack.x
        nimages = stack.info['Dim_1']
        self.__nrows = nimages
        numberofmca = stack.info['Dim_2']
        keylist = ["1.1"] * nimages
        for i in range(nimages):
            keylist[i] = "1.%04d" % i

        for i in range(nimages):
            if self.pleaseBreak:
                break
            self.onImage(keylist[i], keylist)
            self.__ncols = numberofmca
            colsToIter = range(0+self.mcaOffset,
                               numberofmca,
                               self.mcaStep)
            self.__row = i
            self.__col = -1
            try:
                cache_data = data[i, :, :]
            except:
                print("Error reading dataset row %d" % i)
                print(sys.exc_info())
                print("Batch resumed")
                continue
            for mca in colsToIter:
                if self.pleaseBreak:
                    break
                self.__col = mca
                mcadata = cache_data[mca, :]
                y0 = numpy.array(mcadata)
                if xStack is None:
                    if 'MCA start ch' in info:
                        xmin = float(info['MCA start ch'])
                    else:
                        xmin = 0.0
                    x = numpy.arange(len(y0))*1.0 + xmin
                else:
                    x = xStack
                #key = "%s.%s.%02d.%02d" % (scan,order,row,col)
                key = "%s.%04d" % (keylist[i], mca)
                #I only process the first file of the stack?
                filename = os.path.basename(info['SourceName'][0])
                infoDict = {}
                infoDict['SourceName'] = info['SourceName']
                infoDict['Key'] = key
                if "McaLiveTime" in info:
                    infoDict["McaLiveTime"] = \
                            info["McaLiveTime"][i * numberofmca + mca]
                self.__processOneMca(x, y0, filename, key, info=infoDict)
                self.onMca(mca, numberofmca, filename=filename,
                                            key=key,
                                            info=infoDict)

    def __processOneFile(self):
        ffile = self.file
        fileinfo = ffile.GetSourceInfo()
        if 1:
            i = 0
            for scankey in fileinfo['KeyList']:
                if self.pleaseBreak:
                    break
                self.onImage(scankey, fileinfo['KeyList'])
                scan, order = scankey.split(".")
                info, data = ffile.LoadSource(scankey)
                if info['SourceType'] == "EdfFile":
                    nrows = int(info['Dim_1'])
                    ncols = int(info['Dim_2'])
                    numberofmca = ncols
                    self.__ncols = len(range(0+self.mcaOffset,
                                             numberofmca,
                                             self.mcaStep))
                    self.__col = -1
                    for mca_index in range(self.__ncols):
                        mca = 0 + self.mcaOffset + mca_index * self.mcaStep
                        if self.pleaseBreak:
                            break
                        self.__col += 1
                        mcadata = data[mca,:]
                        if 'MCA start ch' in info:
                            xmin = float(info['MCA start ch'])
                        else:
                            xmin = 0.0
                        key = "%s.%s.%04d" % (scan,order,mca)
                        y0  = numpy.array(mcadata)
                        x = numpy.arange(len(y0))*1.0 + xmin
                        filename = os.path.basename(info['SourceName'])
                        infoDict = {}
                        infoDict['SourceName'] = info['SourceName']
                        infoDict['Key']        = key
                        infoDict['McaLiveTime'] = info.get('McaLiveTime', None)
                        self.__processOneMca(x,y0,filename,key,info=infoDict)
                        self.onMca(mca, numberofmca, filename=filename,
                                                    key=key,
                                                    info=infoDict)
                else:
                    if info['NbMca'] > 0:
                        self._initOutputBuffer()
                        numberofmca = info['NbMca'] * 1
                        self.__ncols = len(range(0+self.mcaOffset,
                                             numberofmca,self.mcaStep))
                        numberOfMcaToTakeFromScan = self.__ncols * 1
                        self.__col   = -1
                        scan_key = "%s.%s" % (scan,order)
                        scan_obj= ffile.Source.select(scan_key)
                        #I assume always same number of detectors and
                        #same offset for each detector otherways I would
                        #slow down everything to deal with not very common
                        #situations
                        #if self.__row == 0:
                        if self.counter == 0:
                            self.__chann0List = numpy.zeros(info['NbMcaDet'])
                            chan0list = scan_obj.header('@CHANN')
                            if len(chan0list):
                                for i in range(info['NbMcaDet']):
                                    self.__chann0List[i] = int(chan0list[i].split()[2])
                            # The calculation of self.__ncols is wrong if there are
                            # several scans containing MCAs. One needs to multiply by
                            # the number of scans assuming all of them contain MCAs.
                            # We have to assume the same structure in all files.
                            # Only in the case of "pseudo" two scan files where only
                            # the second scan contains MCAs we do not multiply.
                            if (len(fileinfo['KeyList']) == 2) and (fileinfo['KeyList'].index(scan_key) == 1):
                                # leave self.__ncols untouched
                                self.__ncolsModified = False
                            else:
                                # multiply by the number of scans
                                self.__ncols *= len(fileinfo['KeyList'])
                                self.__ncolsModified = True

                        #import time
                        for mca_index in range(numberOfMcaToTakeFromScan):
                            i = 0 + self.mcaOffset + mca_index * self.mcaStep
                            #e0 = time.time()
                            if self.pleaseBreak: break
                            if self.__ncolsModified:
                                self.__col = i + \
                                      fileinfo['KeyList'].index(scan_key) * \
                                      numberofmca
                            else:
                                self.__col += 1
                            point = int(i/info['NbMcaDet']) + 1
                            mca   = (i % info['NbMcaDet'])  + 1
                            key = "%s.%s.%05d.%d" % (scan,order,point,mca)
                            autotime = self.mcafit.config["concentrations"].get(\
                                        "useautotime", False)
                            if autotime:
                                #slow info reading methods needed to access time
                                mcainfo,mcadata = ffile.LoadSource(key)
                                info['McaLiveTime'] = mcainfo.get('McaLiveTime',
                                                              None)
                            else:
                                mcadata = scan_obj.mca(i+1)
                            y0  = numpy.array(mcadata)
                            x = numpy.arange(len(y0))*1.0 + \
                                self.__chann0List[mca-1]
                            filename = os.path.basename(info['SourceName'])

                            infoDict = {}
                            infoDict['SourceName'] = info['SourceName']
                            infoDict['Key']        = key
                            infoDict['McaLiveTime'] = info.get('McaLiveTime',
                                                               None)
                            self.__processOneMca(x,y0,filename,key,info=infoDict)
                            self.onMca(i, info['NbMca'],filename=filename,
                                                    key=key,
                                                    info=infoDict)
                            #print "remaining = ",(time.time()-e0) * (info['NbMca'] - i)

    def __getFitFile(self, filename, key, createdirs=False):
        fitdir = self.os_path_join(self.outputdir, "FIT")
        if createdirs:
            if not os.path.exists(fitdir):
                try:
                    os.mkdir(fitdir)
                except:
                    print("I could not create directory %s" % fitdir)
                    return
        fitdir = self.os_path_join(fitdir, filename+"_FITDIR")
        if createdirs:
            if not os.path.exists(fitdir):
                try:
                    os.mkdir(fitdir)
                except:
                    print("I could not create directory %s" % fitdir)
                    return
            if not os.path.isdir(fitdir):
                print("%s does not seem to be a valid directory" % fitdir)
                return
        fitfilename = filename + "_" + key + ".fit"
        fitfilename = self.os_path_join(fitdir, fitfilename)
        return fitfilename

    def __getFitConcFile(self):
        if self.chunk is not None:
            con_extension = "_%06d_partial_concentrations.txt" % self.chunk
        else:
            con_extension = "_concentrations.txt"
        cfitfilename = self.os_path_join(self.outputdir,
                                self._rootname + con_extension)
        if self.counter == 0:
            if os.path.exists(cfitfilename):
                try:
                    os.remove(cfitfilename)
                except:
                    print("I could not delete existing concentrations file %s" %\
                        cfitfilename)
        return cfitfilename

    def os_path_join(self, a, b):
        try:
            outfile=os.path.join(a, b)
        except UnicodeDecodeError:
            toBeDone = True
            if sys.platform == 'win32':
                try:
                    outfile=os.path.join(a.decode('mbcs'),
                                         b.decode('mbcs'))
                    toBeDone = False
                except UnicodeDecodeError:
                    pass
            if toBeDone:
                try:
                    outfile = os.path.join(a.decode('utf-8'),
                                           a.decode('utf-8'))
                except UnicodeDecodeError:
                    outfile = os.path.join(a.decode('latin-1'),
                                           a.decode('latin-1'))
        return outfile

    def __processOneMca(self,x,y,filename,key,info=None):
        if self.roiFit:
            result = self.__roiOneMca(x,y)
            self._obsoleteOutputRoiFit(result, filename)  # TODO: remove
            if self.outbuffer and self.__ncols and not self.counter:
                self._allocateMemoryRoiFit(result)
            self._saveRoiFitResult(result)
        else:
            result, concentrations = self.__fitOneMca(x,y,filename,key,info=info)
            self._obsoleteOutputFit(result, concentrations, filename)  # TODO: remove
            if self.outbuffer and self.__ncols and not self.counter:
                self._allocateMemoryFit(result, concentrations)
            self._saveFitResult(result, concentrations)
        self.counter += 1

    def __fitOneMca(self,x,y,filename,key,info=None):
        fitresult = None
        result = None
        concentrations = None
        concentrationsInFitFile = False

        # Fit MCA
        fitfile = self.__getFitFile(filename,key,createdirs=False)
        if os.path.exists(fitfile) and not self.overwrite:
            # Load MCA data when needed
            if outbuffer.saveDiagnostics:
                if not self._attemptMcaLoad(x, y, filename, info=info):
                    return
            # Load result from FIT file
            try:
                fitdict = ConfigDict.ConfigDict()
                fitdict.read(fitfile)
                concentrations = fitdict.get('concentrations', None)
                concentrationsInFitFile = bool(concentrations)
                result = fitdict['result']
            except:
                print("Error trying to use result file %s" % fitfile)
                print("Please, consider deleting it.")
                print(sys.exc_info())
                return
        else:
            # Load MCA data
            if not self._attemptMcaLoad(x, y, filename, info=info):
                return
            # Fit XRF spectrum
            fitresult, result, concentrations = self._fitMca(filename)

        # Extract/calculate + save concentrations
        if result:
            # TODO: concentrationsInResult, when does this happend and should we pop it????
            concentrationsInResult = 'concentrations' not in result
        else:
            concentrationsInResult = False
        if self._concentrations and concentrationsInResult:
            result, concentrations = self._concentrationsFromResult(fitresult, result)
        if self.fitConcFile and concentrations is not None and not concentrationsInFitFile:
            self._updateConcFile(concentrations, filename, key)

        # Digest fit result when not already digested
        if self.fitFiles:
            # Create/update existing FIT file
            fitfile = self.__getFitFile(filename, key, createdirs=True)
            if fitresult:  # TODO: why not "and result is None"?
                result = self.mcafit.digestresult(outfile=fitfile,
                                                  info=info)
            if fitfile:
                if concentrations and not concentrationsInFitFile:
                    self._updateFitFile(concentrations, fitfile)
                self._updateFitFileList(fitfile)
        else:
            if fitresult and result is None:
                # Use imagingDigestResult instead of digestresult:
                # faster and digestresult is not needed just for imaging
                result = self.mcafit.imagingDigestResult()

        return result, concentrations

    def _attemptMcaLoad(self, x, y, filename, info=None):
        try:
            #I make sure I take the fit limits configuration
            self.mcafit.config['fit']['use_limit'] = 1  # TODO: why???
            self.mcafit.setData(x, y, time=info.get("McaLiveTime", None))
        except:
            self._restoreFitConfig(filename)
            return False
        return True

    def _restoreFitConfig(self, filename):
        print("Error entering data of file with output = %s\n%s" %\
                    (filename, sys.exc_info()[1]))
        # Restore when a fit strategy like `matrix adjustment` is used
        if self.mcafit.config['fit'].get("strategyflag", False):
            config = self.__configList[self.__currentConfig]
            print("Restoring fitconfiguration")
            self.mcafit = ClassMcaTheory.McaTheory(config)
            self.mcafit.enableOptimizedLinearFit()  # TODO: why???

    def _fitMca(self, filename):
        result = None
        concentrations = None
        fitresult = None
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
                    conf = self.mcafit.configure()
                    tconf = self._tool.configure()
                    if 'concentrations' in conf:
                        tconf.update(conf['concentrations'])
                    else:
                        #what to do?
                        pass
                    concentrations = self._tool.processFitResult(config=tconf,
                                    fitresult=fitresult0,
                                    elementsfrommatrix=False,
                                    fluorates = self.mcafit._fluoRates)
                except:
                    concentrations = None
                    print("error in concentrations")
                    print(sys.exc_info()[0:-1])
            else:
                #just images
                fitresult = self.mcafit.startfit(digest=0)
        except:
            self._restoreFitConfig(filename)
        return fitresult, result, concentrations

    def _concentrationsFromResult(self, fitresult, result):
        if fitresult:
            fitresult0 = {}
            if result is None:
                result = self.mcafit.digestresult()
            fitresult0['result'] = result
            fitresult0['fitresult'] = fitresult
            conf = self.mcafit.configure()
        else:
            fitresult0 = {}
            fitresult0['result'] = result
            conf = result['config']
        tconf = self._tool.configure()
        if 'concentrations' in conf:
            tconf.update(conf['concentrations'])
        else:
            pass
            #print "Concentrations not calculated"
            #print "Is your fit configuration file correct?"
        try:
            concentrations = self._tool.processFitResult(config=tconf,
                            fitresult=fitresult0,
                            elementsfrommatrix=False)
        except:
            print("error in concentrations")
            print(sys.exc_info()[0:-1])
        return result, concentrations

    def _updateFitFile(self, concentrations, outfile):
        """Add concentrations to fit file
        """
        try:
            f = ConfigDict.ConfigDict()
            f.read(outfile)
            f['concentrations'] = concentrations
            try:
                os.remove(outfile)
            except:
                print("error deleting fit file")
            f.write(outfile)
        except:
            print("Error writing concentrations to fit file")
            print(sys.exc_info())

    def _updateFitFileList(self, outfile):
        """Append FIT file to list of FIT files
        """
        if self.counter:
            self.listfile.write(',\n'+outfile)
        else:
            name = self._rootname +"_fitfilelist.py"
            name = self.os_path_join(self.outputdir,name)
            try:
                os.remove(name)
            except:
                pass
            self.listfile=open(name,"w+")
            self.listfile.write("fitfilelist = [")
            self.listfile.write('\n'+outfile)

    def _updateConcFile(self, concentrations, filename, key):
        if not self.fitConcFile or concentrations is None:
            return
        concentrationsAsAscii = self._toolConversion.getConcentrationsAsAscii(concentrations)
        if len(concentrationsAsAscii) > 1:
            text  = ""
            text += "SOURCE: "+ filename +"\n"
            text += "KEY: "+key+"\n"
            text += concentrationsAsAscii + "\n"
            f = open(self.__getFitConcFile(),"a")
            f.write(text)
        f.close()

    def __roiOneMca(self,x,y):
        return self.mcafit.roifit(x,y,width=self.roiWidth)

    def _allocateMemoryFit(self, result, concentrations):
        outbuffer = self.outbuffer

        # Fit parameters and their uncertainties
        nFree = len(result['groups'])
        imageShape = self.__nrows, self.__ncols
        paramShape = nFree, self.__nrows, self.__ncols
        dtypeResult = numpy.float32
        outbuffer['parameter_names'] = result['groups']
        outbuffer.allocateMemory('parameters',
                                 shape=paramShape,
                                 dtype=dtypeResult,
                                 attrs={'units':'counts'})
        outbuffer.allocateMemory('uncertainties',
                                 shape=paramShape,
                                 dtype=dtypeResult,
                                 attrs={'units':'counts'})

        # Concentrations
        if self._concentrations:
            if 'mmolar' in concentrations:
                concentration_key = 'molarconcentrations'
                concentration_names = 'molarconcentration_names'
                concentration_attrs = {'units': 'mM'}
            else:
                concentration_key = 'massfractions'
                concentration_names = 'massfraction_names'
                concentration_attrs = {}
            self._concentration_key = concentration_key
            self._concentration_names = concentration_names
            outbuffer[concentration_names] = concentrations['groups']
            layerlist = concentrations['layerlist']
            if len(layerlist) > 1:
                outbuffer[concentration_names] += [(group, layer)
                                                    for group in concentrations['groups']
                                                    for layer in layerlist]
            nConcFree = len(concentrations['groups'])
            paramShape = nConcFree, self.__nrows, self.__ncols
            outbuffer.allocateMemory(concentration_key,
                                     shape=paramShape,
                                     dtype=dtypeResult,
                                     attrs=concentration_attrs)

        # Model ,residuals, chisq ,...
        if outbuffer.saveDiagnostics:
            xdata0 = self.mcafit.xdata0.flatten().astype(numpy.int32)  # channels
            xdata = self.mcafit.xdata.flatten().astype(numpy.int32)  # channels after limits
            stackShape = self.__nrows, self.__ncols, len(xdata0)
            mcaIndex = 2
            iXMin, iXMax = xdata[0], xdata[-1]+1
            nObs = iXMax-iXMin
            outbuffer.allocateMemory('nFreeParameters',
                                     shape=imageShape,
                                     fill_value=nFree,
                                     dtype=numpy.int32)
            outbuffer.allocateMemory('nObservations',
                                     shape=imageShape,
                                     fill_value=nObs,
                                     dtype=numpy.int32)
            outbuffer.allocateMemory('Chisq',
                                     shape=imageShape,
                                     fill_value=-1,
                                     dtype=dtypeResult)
            outaxes = False
            if outbuffer.saveFit:
                fitmodel = outbuffer.allocateH5('model',
                                                nxdata='fit',
                                                shape=stackShape,
                                                dtype=dtypeResult,
                                                chunks=True,
                                                fill_value=0,
                                                attrs={'units':'counts'})
                idx = [slice(None)]*fitmodel.ndim
                idx[mcaIndex] = slice(0, iXMin)
                fitmodel[tuple(idx)] = numpy.nan
                idx[mcaIndex] = slice(iXMax, None)
                fitmodel[tuple(idx)] = numpy.nan
                self._mcaIdx = slice(iXMin, iXMax)
            if outbuffer.saveData:
                outaxes = True
                outbuffer.allocateH5('data',
                                     nxdata='fit',
                                     shape=stackShape,
                                     dtype=dtypeResult,
                                     chunks=True,
                                     attrs={'units':'counts'})
            if outbuffer.saveResiduals:
                outaxes = True
                outbuffer.allocateH5('residuals',
                                     nxdata='fit',
                                     shape=stackShape,
                                     dtype=dtypeResult,
                                     chunks=True,
                                     attrs={'units':'counts'})
            if outaxes:
                # Generic axes
                stackAxesNames = ['dim{}'.format(i) for i in range(len(stackShape))]
                dataAxes = [(name, numpy.arange(n, dtype=dtypeResult), {})
                            for name, n in zip(stackAxesNames, stackShape)]
                mcacfg = result['config']['detector']
                linear = result['config']["fit"]["linearfitflag"]
                if linear or (mcacfg['fixedzero'] and mcacfg['fixedgain']):
                    zero = result['fittedpar'][result['parameters'].index('Zero')]
                    gain = result['fittedpar'][result['parameters'].index('Gain')]
                    xenergy = zero + gain*xdata0
                    stackAxesNames[mcaIndex] = 'energy'
                    dataAxes[mcaIndex] = 'energy', xenergy.astype(dtypeResult), {'units': 'keV'}
                    dataAxes.append(('channels', xdata0.astype(numpy.int32), {}))
                outbuffer['dataAxesUsed'] = tuple(stackAxesNames)
                outbuffer['dataAxes'] = tuple(dataAxes)

    def _saveFitResult(self, result, concentrations):
        outbuffer = self.outbuffer
        if outbuffer is None:
            return
        # Fit parameters and their uncertainties
        output = outbuffer['parameters']
        outputs = outbuffer['uncertainties']
        for i, group in enumerate(outbuffer['parameter_names']):
            output[i, self.__row, self.__col] = result[group]['fitarea']
            outputs[i, self.__row, self.__col] = result[group]['sigmaarea']
        # Concentrations
        if self._concentrations:
            output = outbuffer[self._concentration_key]
            for i, name in enumerate(outbuffer[self._concentration_names]):
                if isinstance(name, tuple):
                    group, layer = name
                    output[i, self.__row, self.__col] = concentrations[layer][self.__conKey][group]
                else:
                    output[i, self.__row, self.__col] = concentrations[self.__conKey][name]
        # Diagnostics: model, residuals, chisq ,...
        if outbuffer.saveDiagnostics:
            outbuffer['Chisq'][self.__row, self.__col] = result['chisq']
            idx = self.__row, self.__col, self._mcaIdx
            if outbuffer.saveFit:
                output = outbuffer['model']
                output[idx] = result['yfit']
            if outbuffer.saveData:
                output = outbuffer['data']
                output[idx] = result['ydata']
            if outbuffer.saveResiduals:
                output = outbuffer['residuals']
                output[idx] = result['yfit'] - result['ydata']

    def _obsoleteOutputFit(self, result, concentrations, filename):
        if self.outbuffer is None:
            return
        if self.__ncols is not None:
            if not self.counter:
                self.__peaks  = []
                self.__images = {}
                self.__sigmas = {}
                if not self.__stack:
                    self.__nrows   = len(range(0, len(self._filelist), self.fileStep))
                for group in result['groups']:
                    self.__peaks.append(group)
                    self.__images[group]= numpy.zeros((self.__nrows,
                                                        self.__ncols),
                                                        numpy.float)
                    self.__sigmas[group]= numpy.zeros((self.__nrows,
                                                        self.__ncols),
                                                        numpy.float)
                self.__images['chisq']  = numpy.zeros((self.__nrows,
                                                        self.__ncols),
                                                        numpy.float) - 1.

                if self._concentrations:
                    layerlist = concentrations['layerlist']
                    if 'mmolar' in concentrations:
                        self.__conLabel = " mM"
                        self.__conKey   = "mmolar"
                    else:
                        self.__conLabel = " mass fraction"
                        self.__conKey   = "mass fraction"
                    for group in concentrations['groups']:
                        key = group+self.__conLabel
                        self.__concentrationsKeys.append(key)
                        self.__images[key] = numpy.zeros((self.__nrows,
                                                            self.__ncols),
                                                            numpy.float)
                        if len(layerlist) > 1:
                            for layer in layerlist:
                                key = group+" "+layer
                                self.__concentrationsKeys.append(key)
                                self.__images[key] = numpy.zeros((self.__nrows,
                                                            self.__ncols),
                                                            numpy.float)

        for peak in self.__peaks:
            try:
                self.__images[peak][self.__row, self.__col] = result[peak]['fitarea']
                self.__sigmas[peak][self.__row, self.__col] = result[peak]['sigmaarea']
            except:
                pass
        if self._concentrations:
            layerlist = concentrations['layerlist']
            for group in concentrations['groups']:
                self.__images[group+self.__conLabel][self.__row, self.__col] = \
                                        concentrations[self.__conKey][group]
                if len(layerlist) > 1:
                    for layer in layerlist:
                        self.__images[group+" "+layer] [self.__row, self.__col] = \
                                        concentrations[layer][self.__conKey][group]
        try:
            self.__images['chisq'][self.__row, self.__col] = result['chisq']
        except:
            print("Error on chisq row %d col %d" %\
                    (self.__row, self.__col))
            print("File = %s\n" % filename)
            pass

    def _allocateMemoryRoiFit(self, result):
        outbuffer = self.outbuffer
        parameter_names = [(group, roi)
                           for group, rois in result.items()
                           for roi in rois]
        nFree = len(parameter_names)
        paramShape = nFree, self.__nrows, self.__ncols
        dtypeResult = numpy.float32
        outbuffer['parameter_names'] = parameter_names
        outbuffer.allocateMemory('parameters',
                                    shape=paramShape,
                                    dtype=dtypeResult,
                                    attrs={'units':'counts'})

    def _saveRoiFitResult(self, result):
        outbuffer = self.outbuffer
        if outbuffer is None:
            return
        output = outbuffer['parameters']
        for i, name in enumerate(outbuffer['parameter_names']):
            group, roi = name
            output[i, self.__row, self.__col] = result[group][roi]

    def _obsoleteOutputRoiFit(self, result, filename):
        if self.outbuffer is None:
            return
        if self.__ncols is not None:
            if not self.counter:
                self.__ROIpeaks = []
                self._ROIimages = {}
                if not self.__stack:
                    self.__nrows = len(self._filelist)
                for group in result.keys():
                    self.__ROIpeaks.append(group)
                    self._ROIimages[group]={}
                    for roi in result[group].keys():
                        self._ROIimages[group][roi]=numpy.zeros((self.__nrows,
                                                            self.__ncols),
                                                            numpy.float)

        if not hasattr(self, "_ROIimages"):
            print("ROI fitting only supported on EDF")
        for group in self.__ROIpeaks:
            for roi in self._ROIimages[group].keys():
                try:
                    self._ROIimages[group][roi][self.__row, self.__col] = result[group][roi]
                except:
                    print("error on (row,col) = %d,%d" %\
                            (self.__row, self.__col))
                    print("File = %s" % filename)
                    pass

    def _obsoleteSaveImage(self,ffile=None):
        self.savedImages=[]
        if ffile is None:
            ffile = self.os_path_join(self.imgDir, self._rootname)
        suffix = self._outputSuffix()
        if not self.roiFit:
            #speclabel = "#L row  column"
            speclabel = "row  column"
            iterationList = self.__peaks * 1
            iterationList += ['chisq']
            if self._concentrations:
                iterationList += self.__concentrationsKeys
            for peak in iterationList:
                if peak in self.__peaks:
                    a,b = peak.split()
                    speclabel +="  %s" % (a+"-"+b)
                    speclabel +="  s(%s)" % (a+"-"+b)
                    edfname = ffile +"_"+a+"_"+b+suffix+".edf"
                elif peak in self.__concentrationsKeys:
                    speclabel +="  %s" % peak.replace(" ","-")
                    edfname = ffile +"_"+peak.replace(" ","_")+suffix+".edf"
                elif peak == 'chisq':
                    speclabel +="  %s" % (peak)
                    edfname = ffile +"_"+peak+suffix+".edf"
                else:
                    print("Unhandled peak name: %s. Not saved." % peak)
                    continue
                dirname = os.path.dirname(edfname)
                if not os.path.exists(dirname):
                    try:
                        os.mkdir(dirname)
                    except:
                        print("I could not create directory %s" % dirname)
                Append = 0
                if os.path.exists(edfname):
                    try:
                        os.remove(edfname)
                    except:
                        print("I cannot delete output file")
                        print("trying to append image to the end")
                        Append = 1
                edfout   = EdfFile.EdfFile(edfname, access='ab')
                edfout.WriteImage ({'Title':peak} , self.__images[peak], Append=Append)
                edfout = None
                self.savedImages.append(edfname)
            #save specfile format
            specname = ffile+suffix+".dat"
            if os.path.exists(specname):
                try:
                    os.remove(specname)
                except:
                    pass
            specfile=open(specname,'w+')
            #specfile.write('\n')
            #specfile.write('#S 1  %s\n' % (file+suffix))
            #specfile.write('#N %d\n' % (len(self.__peaks)+2))
            specfile.write('%s\n' % speclabel)
            specline=""
            imageRows = self.__images['chisq'].shape[0]
            imageColumns = self.__images['chisq'].shape[1]
            for row in range(imageRows):
                for col in range(imageColumns):
                    specline += "%d" % row
                    specline += "  %d" % col
                    for peak in self.__peaks:
                        #write area
                        specline +="  %g" % self.__images[peak][row][col]
                        #write sigma area
                        specline +="  %g" % self.__sigmas[peak][row][col]
                    #write global chisq
                    specline +="  %g" % self.__images['chisq'][row][col]
                    if self._concentrations:
                        for peak in self.__concentrationsKeys:
                            specline +="  %g" % self.__images[peak][row][col]
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
                    edfname = ffile+"_"+grouptext+suffix+'.edf'
                    dirname = os.path.dirname(edfname)
                    if not os.path.exists(dirname):
                        try:
                            os.mkdir(dirname)
                        except:
                            print("I could not create directory %s" % dirname)
                    edfout  = EdfFile.EdfFile(edfname)
                    edfout.WriteImage ({'Title':group+" "+roi} , self._ROIimages[group][roi],
                                         Append=i)
                    if i==0:
                        self.savedImages.append(edfname)
                        i=1


if __name__ == "__main__":
    import getopt
    options     = 'f'
    longoptions = ['cfg=', 'pkm=', 'outdir=', 'roifit=', 'roi=',
                   'roiwidth=', 'concentrations=', 'overwrite=',
                   'outroot=', 'outentry=', 'outprocess=',
                   'edf=', 'h5=', 'csv=', 'tif=',
                   'diagnostics=', 'debug=']
    filelist = None
    cfg = None
    roifit = 0
    roiwidth = 250.
    tif = 0
    edf = 1
    csv = 0
    h5 = 1
    debug = 0
    outputDir = None
    concentrations = 0
    saveFit = 0
    saveResiduals = 0
    saveData = 0
    overwrite = 1
    outputRoot = ""
    fileEntry = ""
    fileProcess = ""
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)
    for opt,arg in opts:
        if opt in ('--pkm','--cfg'):
            cfg = arg
        elif opt in ('--outdir'):
            outputDir = arg
        elif opt in ('--roi','--roifit'):
            roifit   = int(arg)
        elif opt in ('--roiwidth'):
            roiwidth = float(arg)
        elif opt in ('--tif', '--tiff'):
            tif = int(arg)
        elif opt == '--edf':
            edf = int(arg)
        elif opt == '--csv':
            csv = int(arg)
        elif opt == '--h5':
            h5 = int(arg)
        elif opt == '--overwrite':
            overwrite = int(arg)
        elif opt == '--concentrations':
            concentrations = int(arg)
        elif opt == '--outroot':
            outputRoot = arg
        elif opt == '--outentry':
            fileEntry = arg
        elif opt == '--outprocess':
            fileProcess = arg
        elif opt == '--debug':
            debug = int(arg)
        elif opt == '--diagnostics':
            saveFit = int(arg)
            saveResiduals = saveFit
            saveData = saveFit
    filelist=args
    if len(filelist) == 0:
        print("No input files, run GUI")
        sys.exit(0)
    t0 = time.time()

    outbuffer = OutputBuffer(outputDir=outputDir,
                        outputRoot=outputRoot, fileEntry=fileEntry,
                        fileProcess=fileProcess, saveData=saveData,
                        saveFit=saveFit, saveResiduals=saveResiduals,
                        tif=tif, edf=edf, csv=csv, h5=h5, overwrite=overwrite)

    from PyMca5.PyMcaMisc import ProfilingUtils
    with ProfilingUtils.profile(memory=debug, time=debug):
        with outbuffer.saveContext():
            b = McaAdvancedFitBatch(cfg,filelist=filelist,
                            fitfiles=1, fitconcfile=1,
                            outputdir=outputDir,
                            roifit=roifit,
                            roiwidth=roiwidth,
                            concentrations=concentrations,
                            outbuffer=outbuffer,
                            overwrite=overwrite)
            b.processList()
        # Without saveContext you need to execute: b.outbuffer.save()
        print("Total Elapsed = % s " % (time.time() - t0))
