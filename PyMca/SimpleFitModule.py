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
import numpy
import sys
import os
import types
import Gefit
import SpecfitFuns

DEBUG = 0

class SimpleFit:
    def __init__(self):
        #get default configuration
        self.getDefaultConfiguration()

        #the list and dictionary of defined functions
        self._functionList = []
        self._functionDict = {}

        #the current fit function
        self._fitFunction = None
        self._stripFunction = None
        self._backgroundFunction = None

    def getDefaultConfiguration(self):
        self._fitConfiguration = {}
        self._fitConfiguration['fit'] = {}
        self._fitConfiguration['fit']['fit_function'] = None
        self._fitConfiguration['fit']['function_flag'] = 1
        self._fitConfiguration['fit']['background_function'] = None
        self._fitConfiguration['fit']['background_flag'] = 1
        self._fitConfiguration['fit']['stripalgorithm'] = "Strip"
        self._fitConfiguration['fit']['strip_flag'] = 1        
        self._fitConfiguration['fit']['fit_algorithm'] = "Levenberg-Marquardt"
        self._fitConfiguration['fit']['weight'] = "NO Weight"
        self._fitConfiguration['fit']['maximum_fit_iterations'] = 10
        self._fitConfiguration['fit']['background_estimation_policy'] = "Estimate always"
        self._fitConfiguration['fit']['function_estimation_policy'] = "Estimate always"
        self._fitConfiguration['fit']['minimum_delta_chi'] = 0.0010
        self._fitConfiguration['fit']['use_limits'] = 0
        self._fitConfiguration['fit']['xmin'] =    0.
        self._fitConfiguration['fit']['xmax'] = 1000.
        self._fitConfiguration['fit']['functions'] = []
        #strip/snip background configuration related
        self._fitConfiguration['fit']['stripanchorsflag'] = 0
        self._fitConfiguration['fit']['stripfilterwidth'] = 1
        self._fitConfiguration['fit']['stripwidth'] = 4
        self._fitConfiguration['fit']['stripiterations'] = 5000
        self._fitConfiguration['fit']['stripconstant'] = 1.0
        self._fitConfiguration['functions'] = {}
        
    def configure(self, ddict):
        print "configuration to be implemented"

    def setData(self, x, y, sigma=None, xmin=None, xmax=None):
        idx = numpy.argsort(x)
        if sigma is not None:
            self._sigma = sigma[idx]
        else:
            self._sigma = None
        self._y0 = y[idx]
        self._x0 = x[idx]
        if sigma is not None:
            self._sigma0 = sigma[idx]
        xmin, xmax = self._getLimits(self._x0, xmin, xmax)
        idx = (self._x0 >= xmin) & (self._x0 <= xmax)
        self._x = self._x0[idx]
        self._y = self._y0[idx]
        if sigma is not None:
            self._sigma = self._sigma0[idx]
        print "TODO: Make sure we have something to fit"
        #get strip/SNIP background
        self._z = self._getStripBackground()

    def importFunctions(self, modname):
        #modname can be a module or a file
        if type(modname) == types.ModuleType:
            newfun = modname
        elif os.path.exists(modname):
            sys.path.append(os.path.dirname(modname))
            f=os.path.basename(os.path.splitext(modname)[0])
            newfun=__import__(f)
            print dir(newfun)
        else:
            raise ValueError, "Cannot interprete/find %s" % modname
            
        theory = newfun.THEORY
        function=newfun.FUNCTION
        parameters = newfun.PARAMETERS
        try:
            estimate=newfun.ESTIMATE
        except:
            estimate=None
        try:
            derivative=newfun.DERIVATIVE
        except:
            derivative=None
        try:
            configure=newfun.CONFIGURE
        except:
            configure=None
        try:
            widget=newfun.WIDGET
        except:
            widget=None

        basename = os.path.basename(newfun.__file__)

        for i in range(len(theory)):
            ddict = {}
            functionName = theory[i]
            ddict['function'] = function[i]
            ddict['parameters'] = parameters[i]
            ddict['estimate']   =None
            ddict['derivative'] =None
            ddict['configure']  =None
            ddict['widget']     =None
            if estimate is not None:
                ddict['estimate'] = estimate[i]
            if derivative is not None:
                ddict['derivative'] = derivative[i]
            if configure is not None:
                ddict['configure'] = configure[i]
            if widget is not None:
                ddict['widget'] = widget[i]
            self._fitConfiguration['functions'][functionName] = ddict
            self._fitConfiguration['fit']['functions'].append(functionName)

    def setFitFunction(self, name):
        self._fitFunctionConfigured = False
        if name not in self._fitConfiguration['fit']['functions']:
            raise KeyError, "Function %s not among defined functions"
        self._fitFunction = name

    def getFitFunction(self):
        return self._fitFunction

    def setBackgroundFunction(self, name):
        if name in [None, "None", "NONE"]:
            self._backgroundFunction = None
            return
        self._backgroundFunctionConfigured = False
        if name not in self._fitConfiguration['fit']['functions']:
            raise KeyError, "Function %s not among defined functions"
        self._backgroundFunction = name

    def getBackgroundFunction(self):
        return self._backgroundFunction

    def _getLimits(self, x, xmin, xmax):
        if self._fitConfiguration['fit']['use_limits']:
            xmin = self._fitConfiguration['fit']['xmin']
            xmax = self._fitConfiguration['fit']['xmax']
            return xmin, xmax
        if xmin is None:
            xmin = x[0]
        if xmax is None:
            xmax = x[-1]
        return xmin, xmax
            
    def _getStripBackground(self, x=None, y=None):
        #this makes the assumption x are equally spaced
        #and I should build a spline if that is not the case
        #but I do not want to put a dependency on SciPy
        if y is not None:
            ywork = y
        else:
            ywork = self._y

        if x is not None:
            xwork = x
        else:
            xwork = self._x

        n=len(xwork)
        #loop for anchors
        anchorslist = []
        if self._fitConfiguration['fit']['stripanchorsflag']:
            if self._fitConfiguration['fit']['stripanchorslist'] is not None:
                oldShape = xwork.shape
                ravelled = xwork
                ravelled.shape = -1
                for channel in self._fitConfiguration['fit']['stripanchorslist']:
                    if channel <= ravelled[0]:continue
                    index = numpy.nonzero(ravelled >= channel)
                    if len(index):
                        index = min(index)
                        if index > 0:
                            anchorslist.append(index)
                ravelled.shape = oldShape

        #work with smoothed data
        ysmooth = self._getSmooth(xwork, ywork)
        
        #SNIP algorithm
        if self._fitConfiguration['fit']['stripalgorithm'] in ["SNIP", 1]:
            if DEBUG:
                print "CALCULATING SNIP"
            if len(anchorslist) == 0:
                anchorslist = [0, len(ysmooth)-1]
            anchorslist.sort()
            result = 0.0 * ysmooth
            lastAnchor = 0
            width = self.config['fit']['snipwidth']
            for anchor in anchorslist:
                if (anchor > lastAnchor) and (anchor < len(ysmooth)):
                    result[lastAnchor:anchor] =\
                            SpecfitFuns.snip1d(ysmooth[lastAnchor:anchor], width, 0)
                    lastAnchor = anchor
            if lastAnchor < len(ysmooth):                
                result[lastAnchor:] =\
                        SpecfitFuns.snip1d(ysmooth[lastAnchor:], width, 0)
            return result
        
        #strip background
        niter = self._fitConfiguration['fit']['stripiterations']
        if niter > 0:
            if DEBUG:
                print "CALCULATING STRIP"
            result=SpecfitFuns.subac(ysmooth,
                                  self._fitConfiguration['fit']['stripconstant'],
                                  niter,
                                  self._fitConfiguration['fit']['stripwidth'],
                                  anchorslist)
            if niter > 1000:
                #make sure to get something smooth
                result = SpecfitFuns.subac(result,
                                  self._fitConfiguration['fit']['stripconstant'],
                                  500,1,
                                  anchorslist)
            else:
                #make sure to get something smooth but with less than
                #500 iterations
                result = SpecfitFuns.subac(result,
                                  self._fitConfiguration['fit']['stripconstant'],
                                  int(self._fitConfiguration['fit']['stripwidth']*2),
                                  1,
                                  anchorslist)
        else:
            result     = numpy.zeros(ysmooth.shape, numpy.float) + min(ysmooth)

        return result

    def _getSmooth(self, x, y): 
        f=[0.25,0.5,0.25]
        try:
            if hasattr(y, "shape"):
                if len(y.shape) > 1:
                    result=SpecfitFuns.SavitskyGolay(numpy.ravel(y).astype(numpy.float), 
                                    self._fitConfiguration['fit']['stripfilterwidth'])
                else:                                
                    result=SpecfitFuns.SavitskyGolay(numpy.array(y).astype(numpy.float), 
                                    self._fitConfiguration['fit']['stripfilterwidth'])
            else:
                result=SpecfitFuns.SavitskyGolay(Numeric.array(y).astype(Numeric.Float), 
                                    self._fitConfiguration['fit']['stripfilterwidth'])
        except Exception, err:
            raise "Error", "Unsuccessful Savitsky-Golay smoothing: %s" % err
            result=numpy.array(y).astype(numpy.float)
        if len(result) > 1:
            result[1:-1]=numpy.convolve(result,f,mode=0)
            result[0]=0.5*(result[0]+result[1])
            result[-1]=0.5*(result[-1]+result[-2])
        return result


        ysmooth = Numeric.ravel(self.__smooth(self.ydata))

        #SNIP algorithm
        if self.config['fit']['stripalgorithm'] == 1:
            if DEBUG:
                print "CALCULATING SNIP"
            if len(anchorslist) == 0:
                anchorslist = [0, len(ysmooth)-1]
            anchorslist.sort()
            self.zz = 0.0 * ysmooth
            lastAnchor = 0
            width = self.config['fit']['snipwidth']
            for anchor in anchorslist:
                if (anchor > lastAnchor) and (anchor < len(ysmooth)):
                    self.zz[lastAnchor:anchor] =\
                            SpecfitFuns.snip1d(ysmooth[lastAnchor:anchor], width, 0)
                    lastAnchor = anchor
            if lastAnchor < len(ysmooth):                
                self.zz[lastAnchor:] =\
                        SpecfitFuns.snip1d(ysmooth[lastAnchor:], width, 0)            
            self.zz.shape = n, 1         
            self.laststripalgorithm  = self.config['fit']['stripalgorithm']
            self.lastsnipwidth       = self.config['fit']['snipwidth']
            self.laststripfilterwidth = self.config['fit']['stripfilterwidth']
            self.laststripanchorsflag     = self.config['fit']['stripanchorsflag']
            self.laststripanchorslist     = self.config['fit']['stripanchorslist']
            return
        
        #strip background
        niter = self.config['fit']['stripiterations']
        if niter > 0:
            if DEBUG:
                print "CALCULATING STRIP"
            if (niter > 1000) and (self.config['fit']['stripwidth'] == 1):
                self.zz=SpecfitFuns.subac(ysmooth,
                                      self.config['fit']['stripconstant'],
                                      niter/20,4, anchorslist)
                self.zz=SpecfitFuns.subac(self.zz,
                                      self.config['fit']['stripconstant'],
                                      niter/4,
                                      self.config['fit']['stripwidth'],
                                      anchorslist)
            else:
                self.zz=SpecfitFuns.subac(ysmooth,
                                      self.config['fit']['stripconstant'],
                                      niter,
                                      self.config['fit']['stripwidth'],
                                      anchorslist)
                if niter > 1000:
                    #make sure to get something smooth
                    self.zz = SpecfitFuns.subac(self.zz,
                                      self.config['fit']['stripconstant'],
                                      500,1,
                                      anchorslist)
                else:
                    #make sure to get something smooth but with less than
                    #500 iterations
                    self.zz = SpecfitFuns.subac(self.zz,
                                      self.config['fit']['stripconstant'],
                                      int(self.config['fit']['stripwidth']*2),
                                      1,
                                      anchorslist)
            self.zz     = Numeric.resize(self.zz,(n,1))
        else:
            self.zz     = Numeric.zeros((n,1),Numeric.Float) + min(ysmooth)

        self.laststripalgorithm  = self.config['fit']['stripalgorithm']
        self.laststripwidth      = self.config['fit']['stripwidth']
        self.laststripfilterwidth = self.config['fit']['stripfilterwidth']
        self.laststripconstant   = self.config['fit']['stripconstant'] 
        self.laststripiterations = self.config['fit']['stripiterations'] 
        self.laststripanchorsflag     = self.config['fit']['stripanchorsflag']
        self.laststripanchorslist     = self.config['fit']['stripanchorslist']


        print "Get strip background to be implemented"
        return numpy.zeros(ywork.shape, numpy.float)

    def fit(self):
        self.estimate()
        self.startFit()
        return self.getResults()

    def estimate(self):
        self._fitResult = None
        self._setStatus("Estimate started")
        backgroundDict  = {'parameters':[]}
        fitFunctionDict = {'parameters':[]}
        backgroundParameters, backgroundConstraints = [], [[],[],[]]
        if self._fitConfiguration['fit']['background_flag']:
            if self._backgroundFunction is not None:
                backgroundParameters, backgroundConstraints =\
                                      self.estimateBackground()
                backgroundDict = self._fitConfiguration['functions']\
                              [self._backgroundFunction]
        self._setStatus("Background estimation finished")
        functionParameters, functionConstraints = [], [[],[],[]]
        if self._fitConfiguration['fit']['function_flag']:
            if self._fitFunction is not None:
                functionParameters, functionConstraints=\
                                    self.estimateFunction()
                fitFunctionDict = self._fitConfiguration['functions']\
                                      [self._fitFunction]
        self._setStatus("Fit function estimation finished")

        #estimations are made
        #Check if there can be conflicts between parameter names
        #because they can have same names in the background and
        #in the fit function
        conflict = False
        for parname in backgroundDict['parameters']:
            if parname in fitFunctionDict['parameters']:
                conflict = True
                break

        #build the parameter names
        self.final_theory=[]
        nBasePar   = len(backgroundDict['parameters'])
        nActualPar = len(backgroundParameters)
        self.__nBackgroundParameters = nActualPar
        if nActualPar:
            for i in range(nActualPar):
                parname = backgroundDict['parameters'][i%nBasePar]
                if conflict:
                    parname = "Bkg_"+parname
                if nBasePar < nActualPar:
                    parname = parname + ("%d" % (int(i/nBasePar)))
                self.final_theory.append(parname)

        nBasePar   = len(fitFunctionDict['parameters'])
        nActualPar = len(functionParameters)
        if nActualPar:
            for i in range(nActualPar):
                parname = fitFunctionDict['parameters'][i%nBasePar]
                if nBasePar < nActualPar:
                    parname = parname + ("%d" % (int(i/nBasePar)))
                self.final_theory.append(parname)

        print self.final_theory 
        CONS=['FREE',
            'POSITIVE',
            'QUOTED',
            'FIXED',
            'FACTOR',
            'DELTA',
            'SUM',
            'IGNORE']
        self.paramlist=[]
        param          = self.final_theory
        j=0
        i=0
        k=0
        xmin=self._x.min()
        xmax=self._x.max()
        #print "xmin = ",xmin,"xmax = ",xmax
        for pname in self.final_theory:
            if i < len(backgroundParameters):
                self.paramlist.append({'name':pname,
                        'estimation':backgroundParameters[i],
                        'group':0,
                        'code':CONS[int(backgroundConstraints[0][i])],
                        'cons1':backgroundConstraints[1][i],
                        'cons2':backgroundConstraints[2][i],
                        'fitresult':0.0,
                        'sigma':0.0,
                        'xmin':xmin,
                        'xmax':xmax})
                i=i+1
            else:
                if (j % len(fitFunctionDict['parameters'])) == 0:
                    k=k+1
                if (CONS[int(functionConstraints[0][j])] == "FACTOR") or \
                   (CONS[int(functionConstraints[0][j])] == "DELTA"):
                        functionConstraints[1][j] = functionConstraints[1][j] +\
                                                    len(backgroundParameters)
                self.paramlist.append({'name':pname,
                       'estimation':functionParameters[j],
                       'group':k,
                       'code':CONS[int(functionConstraints[0][j])],
                       'cons1':functionConstraints[1][j],
                       'cons2':functionConstraints[2][j],
                       'fitresult':0.0,
                       'sigma':0.0,
                       'xmin':xmin,
                       'xmax':xmax})
                j=j+1
        self._setStatus("Estimate finished")
        return self.paramlist

    def _setStatus(self, status):
        self.__status = status

    def getStatus(self):
        return self.__status

    def estimateBackground(self):
        fname = self.getBackgroundFunction()
        if fname is None:
            return [],[[],[],[]]
        ddict = self._fitConfiguration['functions'][fname]
        estimateFunction = ddict['estimate']

        parameters, constraints = estimateFunction(self._x, self._y, self._z)
        return parameters, constraints


    def estimateFunction(self):
        fname = self.getFitFunction()
        if fname is None:
            return [],[[],[],[]]
        ddict = self._fitConfiguration['functions'][fname]
        estimateFunction = ddict['estimate']

        parameters, constraints = estimateFunction(self._x, self._y, self._z)
        return parameters, constraints

    def startFit(self):
        self._setStatus("Fit started")
        print "start fit to be implemented"
        param_list = self.final_theory
        length      = len(param_list)
        param_val   = []
        param_constrains   = [[],[],[]]
        flagconstrained=0
        for param in self.paramlist:
            #print param['name'],param['group'],param['estimation']
            param_val.append(param['estimation'])
            if (param['code'] != 'FREE') & (param['code'] != 0) & \
               (param['code'] != 0.0) :
                flagconstrained=1
            param_constrains [0].append(param['code'])
            param_constrains [1].append(param['cons1'])
            param_constrains [2].append(param['cons2'])

        #weight handling
        if self._fitConfiguration['fit']['weight'] in ["NO Weight", 0]:
            weightflag = 0
        else:
            weightflag = 1

        print "STILL TO HANDLE DERIVATIVES"
        model_deriv = None
        if self._fitConfiguration['fit']['strip_flag']:
            y = self._y - self._z
        else:
            y = self._y
        self._fitResult = None
        if not flagconstrained:
            param_constrains = []
        if DEBUG:
            result = Gefit.LeastSquaresFit(self.modelFunction,param_val,
                    xdata=self._x,
                    ydata=y,
                    sigmadata=self._sigma,
                    constrains=param_constrains,
                    weightflag=weightflag,
                    model_deriv=model_deriv,
                    fulloutput=True)
        else:
            try:
                result = Gefit.LeastSquaresFit(self.modelFunction,param_val,
                        xdata=self._x,
                        ydata=y,
                        sigmadata=self._sigma,
                        constrains=param_constrains,
                        weightflag=weightflag,
                        model_deriv=model_deriv,
                        fulloutput=True)
            except:
                text = sys.exc_info()[1]
                if type(text) is not type(" "): 
                    text = text.args
                    if len(text):
                        text = text[0]
                    else:
                        text = ''
                self._setStatus('Fit error : %s' %text)
                raise

        self._fitResult = {}
        self._fitResult['fittedvalues'] = result[0]
        self._fitResult['chisq']        = result[1]
        self._fitResult['sigma_values'] = result[2]
        self._fitResult['niter']        = result[3]
        self._fitResult['lastdeltachi'] = result[4]
        if DEBUG:
            print "Found parameters = ", self._fitResult['fittedvalues']
        i=0
        for param in self.paramlist:
           if param['code'] != 'IGNORE':
              param['fitresult'] = result[0][i]
              param['sigma']= result[2][i]
           i = i + 1
        self._setStatus("Fit finished")           
        print self.paramlist

    def modelFunction(self, pars, t):
        result = 0.0 * t

        nb = self.__nBackgroundParameters
        if nb:
            result += self._fitConfiguration['functions'][self._backgroundFunction]\
                      ['function'](pars, t)
        if len(self.paramlist) > nb:
            result += self._fitConfiguration['functions'][self._fitFunction]\
                      ['function'](pars, t)
        return result

    def getResults(self):
        print " get results to be implemented"

    def evaluateDefinedFunction(self, x=None):
        print "Evaluate defined function to be implemented"
        if x is None:
            x = self._x
        y = numpy.zeros(x.shape, numpy.float)
        y += self.evaluateBackground(x)
        y += self.evaluateFunction(x)
        if self._fitConfiguration['fit']['strip_flag']:
            #If the x is not self._x, how to add the strip?
            try:
                y += self._z
            except:
                print "Cannot add strip background"
        return y 

def test():
    import SpecfitFunctions
    a=SpecfitFunctions.SpecfitFunctions()
    x = numpy.arange(1000).astype(numpy.float)
    p1 = numpy.array([1500,100.,50.0])
    p2 = numpy.array([1500,700.,50.0])
    y = a.gauss(p1, x)+1
    y = y + a.gauss(p2,x)

    fit = SimpleFit()
    fit.importFunctions(SpecfitFunctions)
    fit.setFitFunction('Gaussians')
    #fit.setBackgroundFunction('Gaussians')
    #fit.setBackgroundFunction('Constant')
    fit.setData(x, y)
    fit.fit()
    print "Expected parameters 1500,100.,50.0, 1500,700.,50.0"
    import PyMcaQt as qt
    import Parameters
    a = qt.QApplication(sys.argv)
    qt.QObject.connect(a,qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))
    w =Parameters.Parameters()
    w.fillfromfit(fit.paramlist)
    w.show()
    a.exec_()

if __name__=="__main__":
    DEBUG = 1
    test()
