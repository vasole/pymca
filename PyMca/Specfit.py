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
import numpy.oldnumeric as Numeric
import sys
import os
import SpecfitFuns
from Gefit import LeastSquaresFit
import EventHandler
DEBUG = 0
class Specfit:
    #def __init__(self,x=None,y=None,sigmay=None):
    def __init__(self,parent=None,*vars,**kw):
        self.fitconfig={}
        self.filterlist=[]
        self.filterdict={}
        self.theorydict={}
        self.theorylist=[]
        self.dataupdate=None
        if 'weight' in kw:
            self.fitconfig['WeightFlag']=kw['weight']
        elif 'WeightFlag' in kw:
            self.fitconfig['WeightFlag']=kw['WeightFlag']
        else:
            self.fitconfig['WeightFlag']=0
        if 'mca' in kw:
            self.fitconfig['McaMode']=kw['mca']
        elif 'McaMode' in kw:
            self.fitconfig['McaMode']=kw['McaMode']
        else:
            self.fitconfig['McaMode']=0
        if 'autofwhm' in kw:
            self.fitconfig['AutoFwhm']=kw['autofwhm']
        elif 'AutoFwhm' in kw:
            self.fitconfig['AutoFwhm']=kw['AutoFwhm']
        else:
            self.fitconfig['AutoFwhm']=0
        if 'fwhm' in kw:
            self.fitconfig['FwhmPoints']=kw['fwhm']
        elif 'FwhmPoints' in kw:
            self.fitconfig['FwhmPoints']=kw['FwhmPoints']
        else:
            self.fitconfig['FwhmPoints']=8
        if 'autoscaling' in kw:
            self.fitconfig['AutoScaling']=kw['autoscaling']
        else:
            self.fitconfig['AutoScaling']=0
        if 'Yscaling' in kw:
            self.fitconfig['Yscaling']=kw['Yscaling']
        else:
            self.fitconfig['Yscaling']=1.0
        if 'Sensitivity' in kw:
            self.fitconfig['Sensitivity']=kw['Sensitivity']
        else:
            self.fitconfig['Sensitivity']=2.5
        if 'ResidualsFlag' in kw:
            self.fitconfig['ResidualsFlag']=kw['ResidualsFlag']
        elif 'Residuals' in kw:
            self.fitconfig['ResidualsFlag']=kw['Residuals']
        else:
            self.fitconfig['ResidualsFlag']=0
        if 'eh' in kw:
            self.eh=kw['eh']
        else:
            self.eh=EventHandler.EventHandler()
        if len(self.theorydict.keys()):
            for key in self.theorydict.keys():
                self.theorylist.append(key)
        self.bkgdict={'No Background':[self.bkg_none,[],None],
                      'Constant':[self.bkg_constant,['Constant'],
                                 self.estimate_builtin_bkg],
                      'Linear':[self.bkg_linear,['Constant','Slope'],
                                 self.estimate_builtin_bkg],
                      'Internal':[self.bkg_internal,
                                 ['Curvature','Iterations','Constant'],
                                 self.estimate_builtin_bkg]}
                      #'Square Filter':[self.bkg_squarefilter,
                      #                ['Width','Constant'],
                      #           self.estimate_builtin_bkg]}
        self.bkglist=[]
        if self.bkgdict.keys() !=[]:
            for key in self.bkgdict.keys():
                self.bkglist.append(key)
        self.fitconfig['fitbkg']='No Background'
        self.bkg_internal_oldx=Numeric.array([])
        self.bkg_internal_oldy=Numeric.array([])
        self.bkg_internal_oldpars=[0,0]
        self.bkg_internal_oldbkg=Numeric.array([])
        self.fitconfig['fittheory']=None
        self.xdata0=Numeric.array([],Numeric.Float)       
        self.ydata0=Numeric.array([],Numeric.Float)
        self.sigmay0=Numeric.array([],Numeric.Float)       
        self.xdata=Numeric.array([],Numeric.Float)       
        self.ydata=Numeric.array([],Numeric.Float)
        self.sigmay=Numeric.array([],Numeric.Float)       
        #if (y is not None):
        #    self.setdata(x,y,sigmay)
        self.setdata(*vars,**kw)
            
    def setdata(self,*vars,**kw):
        if 'x' in kw:
            x=kw['x']
        elif len(vars) >1:
            x=vars[0]                    
        else:
            x=None
        if 'y' in kw:
            y=kw['y']
        elif len(vars) > 1:
            y=vars[1]   
        elif len(vars) == 1:
            y=vars[0]              
        else:
            y=None
        if 'sigmay' in kw:
            sigmay=kw['sigmay']
        elif len(vars) >2:
            sigmay=vars[2]                    
        else:
            sigmay=None
        if y is None:
            return 1
        else:
            self.ydata0=Numeric.array(y)
            self.ydata=Numeric.array(y)
            
        if x is None:
            self.xdata0=Numeric.arange(len(self.ydata0))
            self.xdata=Numeric.arange(len(self.ydata0))            
        else:
            self.xdata0=Numeric.array(x)
            self.xdata=Numeric.array(x)
        
        if sigmay is None:
            dummy = Numeric.sqrt(abs(self.ydata0))
            self.sigmay0=Numeric.reshape(dummy + Numeric.equal(dummy,0),self.ydata0.shape)
            self.sigmay=Numeric.reshape(dummy + Numeric.equal(dummy,0),self.ydata0.shape)
        else:
            self.sigmay0=Numeric.array(sigmay)
            self.sigmay=Numeric.array(sigmay)

        if 'xmin' in kw:
            xmin=kw['xmin']
        else:
            if len(self.xdata):
                xmin=min(self.xdata)
        if 'xmax' in kw:
            xmax=kw['xmax']
        else:
            if len(self.xdata):
                xmax=max(self.xdata)

        if len(self.xdata):
            #sort the data
            i1=Numeric.argsort(self.xdata)
            self.xdata=Numeric.take(self.xdata,i1)
            self.ydata=Numeric.take(self.ydata,i1)
            self.sigmay=Numeric.take(self.sigmay,i1)

            #take the data between limits
            i1=Numeric.nonzero((self.xdata >=xmin) & (self.xdata<=xmax))
            self.xdata=Numeric.take(self.xdata,i1)
            self.ydata=Numeric.take(self.ydata,i1)
            self.sigmay=Numeric.take(self.sigmay,i1)
        
        return 0
    
    def filter(self,*vars,**kw):
        if len(vars) >0:
            xwork=vars[0]
        else:
            xwork=self.xdata0
        if len(vars) >1:
            ywork=vars[1]
        else:
            ywork=self.ydata0
        if len(vars) >2:
            sigmaywork=vars[2]
        else:
            sigmaywork=self.sigmay0
        filterstatus=0
        for i in self.filterlist:
            filterstatus += 1
            try:
                xwork,ywork,sigmaywork=self.filterlist[i][0](xwork,
                                                            ywork,
                                                            sigmaywork,
                                                        self.filterlist[i][1],
                                                        self.filterlist[i][2])
            except:
                return filterstatus
        self.xdata=xwork
        self.ydata=ywork
        self.sigmay=sigmaywork
        return filterstatus

    def addfilter(self,filterfun,*vars,**kw):
        addfilterstatus=0
        if 'filtername' in kw:
            filtername=kw['filtername']
        else:
            kw['filtername']="Unknown"
        self.filterlist.append([filterfun,vars,kw])
        return addfilterstatus
        
    def deletefilter(self,*vars,**kw):
        """
        deletefilter(self,*vars,**kw)
        Deletes a filter from self.filterlist
        self.delete(2) just makes del(self.filterlist[2])
        self.delete(filtername='sort') deletes any filter named sort
        """
        if len(self.filterlist) == 0:
            return 100
        varslist=list(vars)
        index=[]
        deleteerror=0
        for item in varslist:
            if (type(item) == type([])) or \
               (type(item) == type(())):
                for item0 in item:
                        try:
                            newindex=int(item0)                         
                        except:
                            deleteerror=1
            else:
                try:
                    newindex=int(item0)                         
                except:
                    deleteerror=1
            if newindex not in index:
                index.append(newindex)
        if 'filtername' in kw:
            newindex=0
            atleast1=0
            for item in self.filterlist:
                if item[2]['filtername'] == kw['filtername']:
                    atleast1=1
                    if newindex not in index:
                        index.append(newindex)
            if atleast1 == 0:
                deleteerror=2
        if deleteerror:
            return deleteerror
        index.sort()
        index.reverse()
        imin=min(index)
        imax=max(index)
        if imin < 0:
            if imin < -len(self.filterlist):
                deleterror = 3
                return deleteerror
        else:
            if imin >= len(self.filterlist):
                deleterror = 4
                return deleteerror
        if imax < 0:
            if imax < -len(self.filterlist):
                deleterror = 3
                return deleteerror
        else:
            if imax >= len(self.filterlist):
                deleterror = 4
                return deleteerror
        for i in index:
                del(self.filterlist[i])
        return 0

    def addtheory(self, *vars, **kw):
        """
        method addtheory(self,*vars,**kw)
        Usage: self.addtheory(theory,function,parameters,estimate)
               or
               self.addtheory(theory=theory,
                              function=function,
                              parameters=parameters,
                              estimate=estimate)
        Input:
            theory:     String with the name describing the function
            function:   The actual function
            parameters: Parameters names ['p1','p2','p3',...]
            estimate:   The estimate function if any
        Output:
            Returns 0 if everything went fine or a positive number in-
            dicating the offending parameter 
        """
        status=0
        if len(vars) > 0:
            varslist=list(vars)
        else:
            varslist=[]
        if 'theory' in kw:
            theory=kw['theory']
        elif len(varslist) > 0:
            theory=varslist[0]
        else:
            return 1
        if 'function' in kw:
            function=kw['function']
        elif len(varslist) > 1:
            function=varslist[1]
        else:
            return 2
        if 'parameters' in kw:
            parameters=kw['parameters']
        elif len(varslist) > 2:
            parameters=varslist[2]
        else:
            return 3
        if 'estimate' in kw:
            estimate=kw['estimate']
        elif len(varslist) > 3:
            estimate=varslist[3]
        else:
            estimate=None
        if 'configure' in kw:
            configure=kw['configure']
        elif len(varslist) > 4:
            configure=varslist[4]
        else:
            configure=None
        if 'derivative' in kw:
            derivative=kw['derivative']
        elif len(varslist) > 5:
            derivative=varslist[5]
        else:
            derivative=None
        
        self.theorydict[theory]=[function,parameters,estimate,configure,derivative]
        if theory not in self.theorylist:
            self.theorylist.append(theory)
        return 0

    def addbackground(self,*vars,**kw):
        """
        method addbackground(self,*vars,**kw)
        Usage: self.addtheory(background,function,parameters,estimate)
               or
               self.addtheory(background=background,
                              function=function,
                              parameters=parameters,
                              estimate=estimate)
        Input:
            background: String with the name describing the function
            function:   The actual function
            parameters: Parameters names ['p1','p2','p3',...]
            estimate:   The estimate function if any
        Output:
            Returns 0 if everything went fine or a positive number in-
            dicating the offending parameter 
        """
        print("addbackground called")
        status=0
        if len(vars) > 0:
            varslist=list(vars)
        else:
            varslist=[]
        if 'background' in kw:
            theory=kw['background']
        elif len(varslist) > 0:
            theory=varslist[0]
        else:
            return 1
        if 'function' in kw:
            function=kw['function']
        elif len(varslist) > 1:
            theory=varslist[1]
        else:
            return 2
        if 'parameters' in kw:
            function=kw['parameters']
        elif len(varslist) > 2:
            theory=varslist[2]
        else:
            return 3
        if 'estimate' in kw:
            estimate=kw['estimate']
        elif len(varslist) > 3:
            estimate=varslist[3]
        else:
            estimate=None
        
        self.bkgdict[background]=[function,parameters,estimate]
        if theory not in self.bkglist:
            self.bkglist.append(theory)
        return 0

    def settheory(self,theory):
        """
        method: settheory(self,theory)
        Usage: self.settheory(theory)
        Input:
            theory: The name of the theory to be used.
                    It has to be one of the keys of self.theorydict
        Output:
            returns 0 if everything went fine
        """
        if theory in self.theorylist:
            self.fitconfig['fittheory']=theory
            self.theoryfun=self.theorydict[self.fitconfig['fittheory']][0]
            self.modelderiv = None
            if len(self.theorydict[self.fitconfig['fittheory']]) > 5:
                if self.theorydict[self.fitconfig['fittheory']][5] is not None:
                    self.modelderiv = self.myderiv                    
            #I should generate a signal here ...
            return 0
        else:
            return 1

    def setbackground(self,theory):
        """
        method: setbackground(self,background)
        Usage: self.setbackground(background)
        Input:
            theory: The name of the background to be used.
                    It has to be one of the keys of self.bkgdict
        Output:
            returns 0 if everything went fine
        """
        
        if theory in self.bkglist:
            self.fitconfig['fitbkg']=theory
            self.bkgfun=self.bkgdict[self.fitconfig['fitbkg']][0]
            #I should generate a signal here ...
            return 0
        else:
            return 1
        
    def fitfunction(self,pars,t):
        nb=len(self.bkgdict[self.fitconfig['fitbkg']][1])
        #print "nb = ",nb 
        #treat differently user and built in functions
        #if self.selected_th in self.conf.theory_list:
        if (0):
            if (nb>0):
                result = self.bkgfun(pars[0:nb],t) + \
                         self.theoryfun(pars[nb:len(pars)],t)
            else:
                result = self.theoryfun(pars,t)
        else:
            nu=len(self.theorydict[self.fitconfig['fittheory']][1])
            niter=(len(pars)-nb)/nu
            u_term=Numeric.zeros(Numeric.shape(t),Numeric.Float)
            if niter > 0:
                for i in range(niter):
                    u_term= u_term+ \
                            self.theoryfun(pars[(nb+i*nu):(nb+(i+1)*nu)],t)
            if (nb>0):
                result = self.bkgfun(pars[0:nb],t) + u_term
            else:
                result = u_term
        
        if self.fitconfig['fitbkg'] == "Square Filter":
            result=result-pars[1]
            return pars[1]+self.squarefilter(result,pars[0])
        else:
            return result
            
    def estimate(self,mcafit=0):
        """
        Fill the parameters entries with an estimation made on the given data.
        """
        self.state = 'Estimate in progress'
        self.chisq=None
        FitStatusChanged=self.eh.create('FitStatusChanged')
        self.eh.event(FitStatusChanged,data={'chisq':self.chisq,
                                             'status':self.state})

        CONS=['FREE',
            'POSITIVE',
            'QUOTED',
            'FIXED',
            'FACTOR',
            'DELTA',
            'SUM',
            'IGNORE']
       
        #make sure data are current
        if self.dataupdate is not None:
            if not mcafit:
                self.dataupdate()

        xx = self.xdata
        yy = self.ydata

        #estimate the background
        esti_bkg=self.estimate_bkg(xx,yy)
        bkg_esti_parameters = esti_bkg[0]
        bkg_esti_constrains = esti_bkg[1]
        try:
            zz = Numeric.array(esti_bkg[2])
        except:
            zz = Numeric.zeros(Numeric.shape(yy),Numeric.Float)
        #added scaling support
        yscaling=1.0
        if 'AutoScaling' in self.fitconfig:
            if self.fitconfig['AutoScaling']:
                yscaling=self.guess_yscaling(y=yy)
            else:
                if 'Yscaling' in self.fitconfig:
                    yscaling=self.fitconfig['Yscaling']
                else:
                    self.fitconfig['Yscaling']=yscaling
        else:
            self.fitconfig['AutoScaling']=0 
            if 'Yscaling' in self.fitconfig:
                yscaling=self.fitconfig['Yscaling']
            else:
                self.fitconfig['Yscaling']=yscaling
                
        #estimate the function
        estimation = self.estimate_fun(xx,yy,zz,xscaling=1.0,yscaling=yscaling)
        fun_esti_parameters = estimation[0]
        fun_esti_constrains = estimation[1]
        #estimations are made
        #build the names
        self.final_theory=[]
        for i in self.bkgdict[self.fitconfig['fitbkg']][1]:
            self.final_theory.append(i)      
        i=0
        j=1
        while (i < len(fun_esti_parameters)):             
             for k in self.theorydict[self.fitconfig['fittheory']][1]: 
                self.final_theory.append(k+"%d" % j)
                i=i+1
             j=j+1      
             
        self.paramlist=[]
        param          = self.final_theory
        j=0
        i=0
        k=0
        xmin=min(xx)
        xmax=max(xx)
        #print "xmin = ",xmin,"xmax = ",xmax
        for pname in self.final_theory:
            if i < len(bkg_esti_parameters):
                self.paramlist.append({'name':pname,
                                       'estimation':bkg_esti_parameters[i],
                                       'group':0,
                                       'code':CONS[int(bkg_esti_constrains[0][i])],
                                       'cons1':bkg_esti_constrains[1][i],
                                       'cons2':bkg_esti_constrains[2][i],
                                       'fitresult':0.0,
                                       'sigma':0.0,
                                       'xmin':xmin,
                                       'xmax':xmax})
                i=i+1
            else:
                if (j % len(self.theorydict[self.fitconfig['fittheory']][1])) == 0:
                    k=k+1
                if (CONS[int(fun_esti_constrains[0][j])] == "FACTOR") or \
                   (CONS[int(fun_esti_constrains[0][j])] == "DELTA"):
                        fun_esti_constrains[1][j] = fun_esti_constrains[1][j] +\
                                                    len(bkg_esti_parameters)
                self.paramlist.append({'name':pname,
                                       'estimation':fun_esti_parameters[j],
                                       'group':k,
                                       'code':CONS[int(fun_esti_constrains[0][j])],
                                       'cons1':fun_esti_constrains[1][j],
                                       'cons2':fun_esti_constrains[2][j],
                                       'fitresult':0.0,
                                       'sigma':0.0,
                                       'xmin':xmin,
                                       'xmax':xmax})
                j=j+1

        self.state = 'Ready to Fit'
        self.chisq=None
        self.eh.event(FitStatusChanged,data={'chisq':self.chisq,
                                             'status':self.state})
        return self.paramlist
        
    def estimate_bkg(self,xx,yy):
        if self.bkgdict[self.fitconfig['fitbkg']][2] is not None:
            return  self.bkgdict[self.fitconfig['fitbkg']][2](xx,yy)
        else:
            return [],[[],[],[]]

    def estimate_fun(self,xx,yy,zz,xscaling=1.0,yscaling=None):
        if self.theorydict[self.fitconfig['fittheory']][2] is not None:  
            return  self.theorydict[self.fitconfig['fittheory']][2](xx,
                                                                    yy,
                                                                    zz,
                                                                    xscaling=xscaling,
                                                                    yscaling=yscaling)
        else:
            return [],[[],[],[]]
        
    def importfun(self,file):
        sys.path.append(os.path.dirname(file))
        try:
            f=os.path.basename(os.path.splitext(file)[0])
            newfun=__import__(f)
        except:
            msg="Error importing module %s" % file
            #tkMessageBox.showerror('Error', msg)
            return 1
        try:
            init = newfun.INIT
            init()
        except:
            pass
        try:
            theory=newfun.THEORY
        except:
            if DEBUG:
                print("No theory name")
            theory="%s" % file
        try:
            parameters=newfun.PARAMETERS
        except:
            #tkMessageBox.showerror('Error',"Missing PARAMETERS list")
            return 1

        try:
            function=newfun.FUNCTION
        except:
            #tkMessageBox.showerror('Error',"Missing FUNCTION")
            return 1
        
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
            
        badluck=0
        if type(theory) == type([]):
            for i in range(len(theory)):
                if derivative is not None:
                    error=self.addtheory(theory[i],
                                 function[i],
                                 parameters[i],
                                 estimate[i],
                                 configure[i],
                                 derivative[i])
                else:
                    error=self.addtheory(theory[i],
                                 function[i],
                                 parameters[i],
                                 estimate[i],
                                 configure[i],
                                 None)                    
                if error:
                    #tkMessageBox.showerror('Error',"Problem implementing user theory")
                    badluck=1
        else:
            error=self.addtheory(theory,function,parameters,estimate,configure,derivative)
            if error:
                    #tkMessageBox.showerror('Error',"Problem implementing user theory")
                    badluck=1
        if badluck:
            print("ERROR IMPORTING")
        return badluck

    def startfit(self,mcafit=0):
        """
        Launch the fit routine
        """
        if self.dataupdate is not None:
            if not mcafit:
                self.dataupdate()
        FitStatusChanged=self.eh.create('FitStatusChanged')
        self.state = 'Fit in progress'
        self.chisq=None
        self.eh.event(FitStatusChanged,data={'chisq':self.chisq,
                                             'status':self.state})

        param_list = self.final_theory
        length      = len(param_list)
        param_val   = []
        param_constrains   = [[],[],[]]
        flagconstrains=0
        for param in self.paramlist:
            #print param['name'],param['group'],param['estimation']
            param_val.append(param['estimation'])
            if (param['code'] != 'FREE') & (param['code'] != 0) & \
               (param['code'] != 0.0) :
                flagconstrains=1
            param_constrains [0].append(param['code'])
            param_constrains [1].append(param['cons1'])
            param_constrains [2].append(param['cons2'])

        data = []
        i = 0
        ywork=self.ydata*1.0
        if self.fitconfig['fitbkg'] == "Square Filter":
            ywork=self.squarefilter(self.ydata,self.paramlist[0]['estimation'])
            
        for xval in self.xdata:
            if self.sigmay is None:
                data.append([xval,ywork[i]])
            else:
                data.append([xval,ywork[i],
                            self.sigmay[i]])
            i = i + 1

        try:
           if flagconstrains != 1:
                found = LeastSquaresFit(self.fitfunction,param_val,data,
                        weightflag=self.fitconfig['WeightFlag'],
                        model_deriv=self.modelderiv)
           else:
                found = LeastSquaresFit(self.fitfunction,param_val,data,                
                    constrains=param_constrains,
                    weightflag=self.fitconfig['WeightFlag'],
                    model_deriv=self.modelderiv)
        except:
        #except 'LinearAlgebraError' :
           text = sys.exc_info()[1]
           #if type(text) is not string._StringType: 
           if type(text) is not type(" "): 
              text = text.args
              if len(text):
                 text = text[0]
              else:
                 text = ''
           self.state = 'Fit error : %s' %text
           #print 'Fit error : %s' %text
           self.eh.event(FitStatusChanged,data={'chisq':self.chisq,
                                             'status':self.state})
           return

        i=0
        for param in self.paramlist:
           if param['code'] != 'IGNORE':
              param['fitresult'] = found[0][i]
              param['sigma']= found[2][i]
           i = i + 1
        self.chisq = found[1]
        self.state = 'Ready'
        self.eh.event(FitStatusChanged,data={'chisq':self.chisq,
                                             'status':self.state})


    def myderiv(self,param0,index,t0):
        nb=len(self.bkgdict[self.fitconfig['fitbkg']][1])
        #nu=len(self.theorydict[self.fitconfig['fittheory']][1])
        if index >= nb:
            if len(self.theorydict[self.fitconfig['fittheory']])  >5:
                if self.theorydict[self.fitconfig['fittheory']][5] is not None:
                    return self.theorydict[self.fitconfig['fittheory']][5] (param0,index-nb,t0)
                else:  
                    return self.num_deriv(param0,index,t0)
            else:
                return self.num_deriv(param0,index,t0)
        else:
            return self.num_deriv(param0,index,t0)

    def num_deriv(self,param0,index,t0):
        #numerical derivative
        x=Numeric.array(t0)
        delta = (param0[index] + Numeric.equal(param0[index],0.0)) * 0.00001
        newpar = param0.__copy__()
        newpar[index] = param0[index] + delta
        f1 = self.fitfunction(newpar, x)
        newpar[index] = param0[index] - delta
        f2 = self.fitfunction(newpar, x)
        return (f1-f2) / (2.0 * delta)
       
    def gendata(self,*vars,**kw):
        if 'x'in kw:
            x=kw['x']
        elif len(vars) >0:
            x=vars[0]                    
        else:
            x=self.xdata
        if 'parameters' in kw:
            paramlist=kw['parameters']
        elif 'paramlist' in kw:
            paramlist=kw['paramlist']
        elif len(vars) >1:
            paramlist=vars[1]                    
        else:
            paramlist=self.paramlist
        noigno = []
        for param in paramlist:
           if param['code'] != 'IGNORE':
              noigno.append(param['fitresult'])

        #next two lines gave problems with internal background after a zoom
        #newdata = self.fit_fun0(take(found[0],noigno).tolist(),Numeric.array(self.xdata0))
        #newdata = newdata * self.mondata0
        newdata = self.fitfunction(noigno,Numeric.array(x))
        return newdata
        
    def bkg_constant(self,pars,x):
        """
        Constant background
        """
        return pars[0]  * Numeric.ones(Numeric.shape(x),Numeric.Float)

    def bkg_linear(self,pars,x):
        """
        Linear background
        """
        return pars[0] + pars [1] * x  

    def bkg_internal(self,pars,x):
        """
        Internal Background
        """
        #fast
        #return self.zz  
        #slow: recalculate the background as function of the parameters
        #yy=SpecfitFuns.subac(self.ydata*self.fitconfig['Yscaling'],
        #                     pars[0],pars[1])
        if self.bkg_internal_oldpars[0] == pars[0]:
          if self.bkg_internal_oldpars[1] == pars[1]:
            if (len(x) == len(self.bkg_internal_oldx)) & \
               (len(self.ydata) == len(self.bkg_internal_oldy)):
                    #same parameters
                    if Numeric.sum(self.bkg_internal_oldx == x) == len(x):
                        if Numeric.sum(self.bkg_internal_oldy == self.ydata) == len(self.ydata):
                            return self.bkg_internal_oldbkg + pars[2] * Numeric.ones(Numeric.shape(x),Numeric.Float)                            
        self.bkg_internal_oldy=self.ydata
        self.bkg_internal_oldx=x
        self.bkg_internal_oldpars=pars
        try:
            idx = Numeric.nonzero((self.xdata>=x[0]) & (self.xdata<=x[-1]))
        except:
            print("ERROR ",x)
        yy=Numeric.take(self.ydata,idx)
        nrx=Numeric.shape(x)[0]
        nry=Numeric.shape(yy)[0]
        if nrx == nry:
            self.bkg_internal_oldbkg=SpecfitFuns.subac(yy,pars[0],pars[1]) 
            return self.bkg_internal_oldbkg + pars[2] * Numeric.ones(Numeric.shape(x),Numeric.Float)                            

        else:
            self.bkg_internal_oldbkg=SpecfitFuns.subac(Numeric.take(yy,Numeric.arange(0,nry,2)),
                                    pars[0],pars[1])
            return self.bkg_internal_oldbkg + pars[2] * Numeric.ones(Numeric.shape(x),Numeric.Float)                            

    def bkg_squarefilter(self,pars,x):
        """
        Square filter Background
        """
        #yy=self.squarefilter(self.ydata,pars[0])
        return pars[1]  * Numeric.ones(Numeric.shape(x),Numeric.Float)
            
    def bkg_none(self,pars,x):
        """
        Internal Background
        """       
        return Numeric.zeros(x.shape,Numeric.Float)

    def estimate_builtin_bkg(self,xx,yy):
       self.zz=SpecfitFuns.subac(yy,1.0001,1000)
       zz = self.zz
       npoints = len(zz)
       if self.fitconfig['fitbkg'] == 'Constant':       
            #Constant background
            S = float(npoints)
            Sy = min(zz)
            fittedpar=[Sy]
            cons = Numeric.zeros((3,len(fittedpar)),Numeric.Float)
       elif self.fitconfig['fitbkg'] == 'Internal':
            #Internal
            fittedpar=[1.000,10000,0.0]
            cons = Numeric.zeros((3,len(fittedpar)),Numeric.Float)
            cons[0][0]= 3
            cons[0][1]= 3
            cons[0][2]= 3
       elif self.fitconfig['fitbkg'] == 'No Background':
            #None
            fittedpar=[]
            cons = Numeric.zeros((3,len(fittedpar)),Numeric.Float)
       elif self.fitconfig['fitbkg'] == 'Square Filter':
            fwhm=5
            if 'AutoFwhm' in self.fitconfig:
                fwhm=self.guess_fwhm(y=y)            
            elif 'fwhm' in self.fitconfig:
                fwhm=self.fitconfig['fwhm']
            elif 'Fwhm' in self.fitconfig:
                fwhm=self.fitconfig['Fwhm']
            elif 'FWHM' in self.fitconfig:
                fwhm=self.fitconfig['FWHM']
            elif 'FwhmPoints' in self.fitconfig:
                fwhm=self.fitconfig['FwhmPoints']
            #set an odd number
            if (fwhm % 2):
                fittedpar=[fwhm,0.0]
            else:
                fittedpar=[fwhm+1,0.0]
            cons = Numeric.zeros((3,len(fittedpar)),Numeric.Float)
            cons[0][0]= 3
            cons[0][1]= 3
       else:
            S = float(npoints)
            Sy = Numeric.sum(zz)
            Sx = float(Numeric.sum(xx))
            Sxx = float(Numeric.sum(xx * xx))            
            Sxy = float(Numeric.sum(xx * zz))
                                
            deno = S * Sxx - (Sx * Sx)
            if (deno != 0):
                bg = (Sxx * Sy - Sx * Sxy)/deno
                slop = (S * Sxy - Sx * Sy)/deno
            else:
                bg = 0.0
                slop = 0.0        
            fittedpar=[bg/1.0,slop/1.0]
            cons = Numeric.zeros((3,len(fittedpar)),Numeric.Float)
       return fittedpar,cons,zz

    def configure(self,**kw):
        """
        Configure the current theory passing a dictionary to the supply method
        """
        for key in self.fitconfig.keys():
            if key in kw:
                self.fitconfig[key]=kw[key]
        result={}
        result.update(self.fitconfig)
        if self.fitconfig['fittheory'] is not None:
         if self.fitconfig['fittheory'] in self.theorydict.keys():
          if self.theorydict[self.fitconfig['fittheory']][3] is not None:
           result.update(self.theorydict[self.fitconfig['fittheory']][3](**kw))
           #take care of possible user interfaces
           for key in self.fitconfig.keys():
                if key in result:
                    self.fitconfig[key]=result[key]
        #make sure fitconfig is configured in case of having the same keys
        for key in self.fitconfig.keys():
            if key in kw:
                self.fitconfig[key]=kw[key]
            if key == "fitbkg":
                error = self.setbackground(self.fitconfig[key])
            if key == "fittheory":
                if self.fitconfig['fittheory'] is not None:
                    error = self.settheory(self.fitconfig[key])
        if error:
            print("ERROR on background and/or theory configuration")
        result.update(self.fitconfig)
        return result 

    def mcafit(self,*vars,**kw):
        if (len(vars) > 0) or (len(kw.keys()) > 0):
            if 'x' in kw:
                x=kw['x']
            elif len(vars) >1:
                x=vars[0]                    
            else:
                x=None
            if 'y' in kw:
                y=kw['y']
            elif len(vars) == 1:
                y=vars[0]
            elif len(vars) > 1:
                y=vars[1]
            else:
                y=self.ydata0
            if 'sigmay' in kw:
                sigmay=kw['sigmay']
            elif len(vars) >2:
                sigmay=vars[2]                    
            else:
                sigmay=None
            if x is None:
                x=Numeric.arange(len(y)).astype(Numeric.Float)
            if sigmay is None:
                self.setdata(x,y,**kw)
            else:
                self.setdata(x,y,sigmay,**kw)
        else:
            #make sure data are current
            if self.dataupdate is not None:
                self.dataupdate()
 
        if 'debug' in kw:
            mcadebug = 1
        else:
            mcadebug = 0
        if 'Yscaling' in kw:
            if kw['Yscaling'] is not None:
                yscaling=kw['Yscaling']
        elif 'yscaling' in kw:
            if kw['yscaling'] is not None:
                yscaling=kw['yscaling']
        elif self.fitconfig['AutoScaling']:
                yscaling=self.guess_yscaling()   
        else:
            yscaling=self.fitconfig['Yscaling']
        
        if 'sensitivity' in kw:
            sensitivity=kw['sensitivity']
        elif 'Sensitivity' in kw:
            sensitivity=kw['Sensitivity']
        else:
            sensitivity=self.fitconfig['Sensitivity']            
        
        if 'fwhm' in kw:
            fwhm=kw['fwhm']
        elif 'FwhmPoints' in kw:
            fwhm=kw['FwhmPoints']
        elif self.fitconfig['AutoFwhm']:
            fwhm=self.guess_fwhm(y=y)
        else:
            fwhm=self.fitconfig['FwhmPoints']
            
        fwhm=int(fwhm)
        #print self.fitconfig['FwhmPoints']
        #print "yscaling    = ",yscaling
        #print "fwhm        = ",fwhm
        #print "sensitivity = ",sensitivity
        #removed this line self.fitconfig['fwhm']=fwhm
        #print "mca yscaling = ",yscaling,"fwhm = ",fwhm
        
        #needed to make sure same peaks are found
        self.configure(Yscaling=yscaling,
                        #lowercase on purpose
                        autoscaling=0,
                        FwhmPoints=fwhm,
                        Sensitivity=sensitivity)
        ysearch=self.ydata*yscaling
        npoints=len(ysearch)
        peaks=[]
        if npoints > (1.5)*fwhm:
            peaksidx=SpecfitFuns.seek(ysearch,0,npoints,
                                    fwhm,
                                    sensitivity)
            for idx in peaksidx:
                peaks.append(self.xdata[int(idx)])
            if mcadebug:
                print("MCA Found peaks = ",peaks)
        if len(peaks):
            regions=self.mcaregions(peaks,self.xdata[fwhm]-self.xdata[0])
        else:
            regions=[]
        if mcadebug:
            print(" regions = ",regions)
        #if the function needs a scaling just give it
        #removed estimate should deal with it 
        #self.configure(Yscaling=yscaling,yscaling=yscaling)
        #removed, configure should deal with it
        #self.configure(fwhm=fwhm,FwhmPoints=fwhm)
        mcaresult=[]
        #import SimplePlot
        xmin0 = self.xdata[0]
        xmax0 = self.xdata[-1]        
        for region in regions:
            if(0):
                idx=Numeric.argsort(self.xdata0)
                self.xdata=Numeric.take(self.xdata0,idx)
                self.ydata=Numeric.take(self.ydata0,idx)
                #self.sigmay=Numeric.take(self.sigmay0,idx)
                idx = Numeric.nonzero((self.xdata>=region[0]) & (self.xdata<=region[1]))
                self.xdata=Numeric.take(self.xdata,idx)
                self.ydata=Numeric.take(self.ydata,idx)
            self.setdata(self.xdata0,self.ydata0,self.sigmay0,xmin=region[0],xmax=region[1])
            #SimplePlot.plot([self.xdata,self.ydata],yname='region to fit')
            if 0:
                #the calling program shoudl take care of sigma
                self.sigmay=Numeric.sqrt(self.ydata/yscaling)
                self.sigmay=self.sigmay+Numeric.equal(self.sigmay,0)
            self.estimate(mcafit=1)
            if self.state == 'Ready to Fit':
              self.startfit(mcafit=1)
              if self.chisq is not None:
               if self.fitconfig['ResidualsFlag']:
                while(self.chisq > 2.5):
                #awful fit, simple residuals search adding a gaussian(?)
                  if (0):
                    error=self.mcaresidualssearch_old()
                    print("error = ",error)
                    if not error:
                        for param in self.paramlist:
                            param['estimation']=param['fitresult']
                        self.startfit()
                  newpar,newcons=self.mcaresidualssearch()
                  if newpar != []:
                    newg=1
                    for param in self.paramlist: 
                        newg=max(newg,int(float(param['group'])+1))
                        param['estimation']=param['fitresult']
                    i=-1                    
                    for pname in self.theorydict[self.fitconfig['fittheory']][1]:
                        i=i+1
                        name=pname + "%d" % newg
                        self.paramlist.append({'name':name,
                                   'estimation':newpar[i],
                                   'group':newg,
                                   'code':newcons[0][i],
                                   'cons1':newcons[1][i],
                                   'cons2':newcons[2][i],
                                   'fitresult':0.0,
                                   'sigma':0.0})
                    self.startfit()
                  else:
                    break
            #import SimplePlot
            #SimplePlot.plot([self.xdata,(self.ydata-self.gendata())/self.sigmay],
            #                xname='X',yname='Norm Residuals')
            mcaresult.append(self.mcagetresult())
            self.setdata(self.xdata0,self.ydata0,xmin=xmin0,xmax=xmax0)
            #result=self.mcagetresult()
            #for (pos,area,sigma_area,mca_fwhm) in result['mca_areas']:
            #        print "Pos = ",pos,"Area = ",area,"sigma_area =",sigma_area

        #for result in mcaresult:
        #    for (pos,area,sigma_area,mca_fwhm) in result['mca_areas']:
        #        print "Pos = ",pos,"Area = ",area,"sigma_area =",sigma_area
        return mcaresult
        
    def mcaregions(self,peaks,fwhm):
        mindelta=3.0*fwhm
        plusdelta=3.0*fwhm
        regions=[]
        xdata0 = min(self.xdata[0], self.xdata[-1])
        xdata1 = max(self.xdata[0], self.xdata[-1])
        for peak in peaks:
            x0=max(peak-mindelta, xdata0)
            x1=min(peak+plusdelta, xdata1)
            if regions == []:
                regions.append([x0,x1])
            else:
                if x0 < regions[-1][0]:
                    regions[-1][0]=x0
                elif x0 < (regions[-1][1]):
                    regions[-1][1]=x1
                else:
                    regions.append([x0,x1])
        return regions
        
    def mcagetresult(self):
        result={}
        result['xbegin']    = min(self.xdata[0], self.xdata[-1])
        result['xend']      = max(self.xdata[0], self.xdata[-1])
        try:
            result['fitstate']  = self.state
        except:
            result['fitstate']  = 'Unknown' 
        result['fitconfig'] = self.fitconfig
        result['config']    = self.configure()
        result['paramlist'] = self.paramlist
        result['chisq']     = self.chisq
        result['mca_areas']=self.mcagetareas()

        return result

    def mcagetareas(self,**kw):
        if 'x' in kw:
            x=kw['x']
        else:
            x=self.xdata
        if 'y' in kw:
            y=kw['y']
        else:
            y=self.ydata
        if 'sigmay' in kw:
            sigmay=kw['sigmay']
        else:
            sigmay=self.sigmay
        if 'parameters' in kw:
            paramlist=kw['parameters']
        elif 'paramlist' in kw:
            paramlist=kw['paramlist']
        else:
            paramlist=self.paramlist
        noigno = []
        groups=[]
        for param in paramlist:
            if param['code'] != 'IGNORE':
                if (int(float(param['group'])) != 0):
                    if param['group'] not in groups:
                        groups.append(param['group'])

        result=[]
        for group in groups:
            noigno=[]
            pos=0
            area=0
            sigma_area=0
            fwhm=0
            for param in paramlist:
                if param['group'] != group:
                    if param['code'] != 'IGNORE':
                        noigno.append(param['fitresult'])
                else:
                    if param['name'].find('Position') != -1:
                        pos=param['fitresult']
                    if (param['name'].find('FWHM') != -1) | \
                       (param['name'].find('Fwhm') != -1) | \
                       (param['name'].find('fwhm') != -1):
                            fwhm=param['fitresult']
            #now I add everything around +/- 4 sigma
            #around the peak position
            sigma=fwhm/2.354
            xmin = max(pos-3.99*sigma,min(x))
            xmax = min(pos+3.99*sigma,max(x))
            #xmin=min(x)
            #xmax=max(x)
            idx = Numeric.nonzero((x>=xmin) & (x<=xmax))
            x_around=Numeric.take(x,idx)
            y_around=Numeric.take(y,idx)
            ybkg_around=Numeric.take(self.fitfunction(noigno,x),idx)
            if 0:
                #only valid for MCA's!!!
                area=(Numeric.sum(y_around-ybkg_around))
            else:
                neto = y_around-ybkg_around
                deltax = x_around[1:] - x_around[0:-1]
                area=Numeric.sum(neto[0:-1]*deltax)
            sigma_area=(Numeric.sqrt(Numeric.sum(y_around)))
            result.append([pos,area,sigma_area,fwhm])           
            #import SimplePlot
            #SimplePlot.plot([Numeric.take(x,idx),y_around,ybkg_around])
            #SimplePlot.plot([x,y,self.fitfunction(noigno,x)],yname='Peak Area')

        return result

    def guess_yscaling(self,*vars,**kw):
        if 'y' in kw:
            y=kw['y']
        elif len(vars) > 0:
            y=vars[0]
        else:
            y=self.ydata
            
        zz=SpecfitFuns.subac(y,1.0,10)        
        if 0:
            idx=Numeric.nonzero(zz>(min(y/100.)))
            yy=Numeric.take(y,idx)
            yfit=Numeric.take(zz,idx)
        elif 1:
                zz=Numeric.convolve(y,[1.,1.,1.])/3.0
                yy=y[1:-1]
                yfit=zz
                idx=Numeric.nonzero(Numeric.fabs(yy)>0.0)
                yy=Numeric.take(yy,idx)
                yfit=Numeric.take(yfit,idx)
        else:
            yy=y
            yfit=zz
        #avoid case of dividing by 0
        try:
            chisq=Numeric.sum(((yy-yfit)*(yy-yfit))/(Numeric.fabs(yy)*len(yy)))
            scaling=1./chisq
        except:
            scaling=1.0
        return scaling

    def guess_fwhm(self,**kw):
        if 'x' in kw:
            x=kw['x']
        else:
            x=self.xdata
        if 'y' in kw:
            y=kw['y']
        else:
            y=self.ydata
        #set at least a default value for the fwhm    
        fwhm=4

        zz=SpecfitFuns.subac(y,1.000,1000)
        yfit=y-zz
        
        #now I should do some sort of peak search ...
        maximum=max(yfit)
        idx=Numeric.nonzero(yfit == maximum)
        pos=Numeric.take(x,idx)[-1]
        posindex=idx[-1]
        height=yfit[posindex]
        imin=posindex
        while ((yfit[imin] > 0.5*height) & (imin >0)):
            imin=imin - 1
        imax=posindex
        while ((yfit[imax] > 0.5*height) & (imax <(len(yfit)-1))):
            imax=imax + 1
        fwhm=max(imax-imin-1,fwhm)
            
        return fwhm            

    def mcaresidualssearch(self,**kw):
        if 'y' in kw:
            y=kw['y']
        else:
            y=self.ydata
        if 'x' in kw:
            x=kw['x']
        else:
            x=self.xdata
        if 'sigmay' in kw:
            sigmay=kw['sigmay']
        else:
            sigmay=self.sigmay
        if 'parameters' in kw:
            paramlist=kw['parameters']
        elif 'paramlist' in kw:
            paramlist=kw['paramlist']
        else:
            paramlist=self.paramlist

        newpar=[]
        newcodes=[[],[],[]]
        if self.fitconfig['fitbkg'] == 'Square Filter':
            y=self.squarefilter(y,paramlist[0]['estimation'])
            return newpar,newcodes
        areanotdone=1
        
        #estimate the fwhm
        fwhm=10
        fwhmcode='POSITIVE'
        fwhmcons1=0
        fwhmcons2=0
        i=-1
        peaks=[]
        for param in paramlist:
            i=i+1
            pname=param['name']
            if (pname.find('FWHM') != -1) | \
               (pname.find('Fwhm') != -1) | \
               (pname.find('fwhm') != -1):
                    fwhm=param['fitresult']
                    if (param['code'] == 'FREE')  | \
                       (param['code'] == 'FIXED') | \
                       (param['code'] == 'QUOTED')| \
                       (param['code'] == 'POSITIVE')| \
                       (param['code'] == 0)| \
                       (param['code'] == 1)| \
                       (param['code'] == 2)| \
                       (param['code'] == 3):
                            fwhmcode='FACTOR'
                            fwhmcons1=i
                            fwhmcons2=1.0
            if pname.find('Position') != -1:
                    peaks.append(param['fitresult'])

        #print "Residuals using fwhm = ",fwhm

        #calculate the residuals
        yfit = self.gendata(x=x,paramlist=paramlist)
            
        residuals=(y-yfit)/(sigmay+Numeric.equal(sigmay,0.0))

        #set to zero all the residuals around peaks
        for peak in peaks:
            idx=Numeric.less(x,peak-0.8*fwhm)+Numeric.greater(x,peak+0.8*fwhm)
            yfit=yfit*idx
            y=y*idx
            residuals=residuals*idx

        
        #estimate the position
        maxres=max(residuals)
        idx=Numeric.nonzero(residuals == maxres)
        pos=Numeric.take(x,idx)[-1]
        
        #estimate the height!
        height=Numeric.take(y-yfit,idx)[-1]
        if (height <= 0):
            return newpar,newcodes
        
        for pname in self.theorydict[self.fitconfig['fittheory']][1]:
            estimation=0.0
            if pname.find('Position') != -1:
                    estimation=pos
                    code='QUOTED'
                    cons1=pos-0.5*fwhm
                    cons2=pos+0.5*fwhm
            elif pname.find('Area')!= -1:
                if areanotdone:                    
                    areanotdone=0
                    area=(height * fwhm / (2.0*Numeric.sqrt(2*Numeric.log(2))))* \
                                Numeric.sqrt(2*Numeric.pi)
                    if area <= 0:
                        return [],[[],[],[]]
                    estimation=area
                    code='POSITIVE'
                    cons1=0.0
                    cons2=0.0
                else:
                    estimation=0.0
                    code='FIXED'
                    cons1=0.0
                    cons2=0.0
            elif (pname.find('FWHM') != -1) | \
                 (pname.find('Fwhm') != -1) | \
                 (pname.find('fwhm') != -1):
                    estimation=fwhm
                    code=fwhmcode
                    cons1=fwhmcons1
                    cons2=fwhmcons2
            else:
                    estimation=0.0
                    code='FIXED'
                    cons1=0.0
                    cons2=0.0
            newpar.append(estimation)
            newcodes[0].append(code)
            newcodes[1].append(cons1)
            newcodes[2].append(cons2)
        return newpar,newcodes

    def mcaresidualssearch_old(self,**kw):
        if 'y' in kw:
            y=kw['y']
        else:
            y=self.ydata
        if 'x' in kw:
            x=kw['x']
        else:
            x=self.xdata
        if 'sigmay' in kw:
            sigmay=kw['sigmay']
        else:
            sigmay=self.sigmay
        if 'parameters' in kw:
            paramlist=kw['parameters']
        elif 'paramlist' in kw:
            paramlist=kw['paramlist']
        else:
            paramlist=self.paramlist
        areanotdone=1
        newg=1
        for param in self.paramlist: 
            newg=max(newg,int(float(param['group'])+1))
        if newg == 1:
            return areanotdone
        
        #estimate the fwhm
        fwhm=10
        fwhmcode='POSITIVE'
        fwhmcons1=0
        fwhmcons2=0
        i=-1
        peaks=[]
        for param in paramlist:
            i=i+1
            pname=param['name']
            if (pname.find('FWHM') != -1) | \
               (pname.find('Fwhm') != -1) | \
               (pname.find('fwhm') != -1):
                    fwhm=param['fitresult']
                    if (param['code'] == 'FREE')  | \
                       (param['code'] == 'FIXED') | \
                       (param['code'] == 'QUOTED')| \
                       (param['code'] == 'POSITIVE')| \
                       (param['code'] == 0)| \
                       (param['code'] == 1)| \
                       (param['code'] == 2)| \
                       (param['code'] == 3):
                            fwhmcode='FACTOR'
                            fwhmcons1=i
                            fwhmcons2=1.0
            if pname.find('Position') != -1:
                    peaks.append(param['fitresult'])

        #calculate the residuals
        yfit = self.gendata(x=x,paramlist=paramlist)
        residuals=(y-yfit)/(sigmay+Numeric.equal(sigmay,0.0))

        #set to zero all the residuals around peaks
        for peak in peaks:
            idx=Numeric.less(x,peak-0.8*fwhm)+Numeric.greater(x,peak+0.8*fwhm)
            yfit=yfit*idx
            y=y*idx
            residuals=residuals*idx

        
        #estimate the position
        maxres=max(residuals)
        idx=Numeric.nonzero(residuals == maxres)
        pos=Numeric.take(x,idx)[-1]


        
        #estimate the height!
        height=Numeric.take(y-yfit,idx)[-1]
        
        for pname in self.theorydict[self.fitconfig['fittheory']][1]:
            estimation=0.0
            name=pname+ "%d" % newg
            self.final_theory.append(pname)
            if pname.find('Position') != -1:
                    estimation=pos
                    code='QUOTED'
                    cons1=pos-0.5*fwhm
                    cons2=pos+0.5*fwhm
            elif pname.find('Area')!= -1:
                if areanotdone:                    
                    areanotdone=0
                    estimation=(height * fwhm / (2.0*Numeric.sqrt(2*Numeric.log(2))))* \
                                Numeric.sqrt(2*Numeric.pi)
                    code='POSITIVE'
                    cons1=0.0
                    cons2=0.0
                else:
                    estimation=0.0
                    code='FIXED'
                    cons1=0.0
                    cons2=0.0
            elif (pname.find('FWHM') != -1) | \
                 (pname.find('Fwhm') != -1) | \
                 (pname.find('fwhm') != -1):
                    estimation=fwhm
                    code=fwhmcode
                    cons1=fwhmcons1
                    cons2=fwhmcons2
            else:
                    estimation=0.0
                    code='FIXED'
                    cons1=0.0
                    cons2=0.0
            paramlist.append({'name':pname,
                                   'estimation':estimation,
                                   'group':newg,
                                   'code':code,
                                   'cons1':cons1,
                                   'cons2':cons2,
                                   'fitresult':0.0,
                                   'sigma':0.0})
        return areanotdone

    def numderiv(self,*vars,**kw):
        """
        numeriv(self,*vars,**kw)
        Usage: self.numderiv(x,y)
               self.numderiv(x=x,y=y)
               self.numderiv()
        """
        if 'y' in kw:
            ydata=kw['y']
        elif len(vars) > 1:
            ydata=vars[1]
        else:
            ydata=self.y
        if 'x' in kw:
            xdata=kw['x']
        elif len(vars) > 0:
            xdata=vars[0]
        else:
            xdata=self.x
        f=[1,-1]
        x=Numeric.array(xdata)
        y=Numeric.array(ydata)
        x,y = self.pretreat(x,y)
        deltax=Numeric.convolve(x,f,mode=0)
        i1=Numeric.nonzero(abs(deltax)>0.0000001)
        deltay=Numeric.convolve(y,f,mode=0)
        deno=Numeric.take(deltax,i1)
        num=Numeric.take(deltay,i1)
        #Still what to do with the first and last point ...
        try:
            derivfirst=Numeric.array((y[1]-y[0])/(x[1]-x[0]))
        except:
            derivfirst=Numeric.array([])
        try:
            derivlast= Numeric.array((y[-1]-y[-2])/(x[-1]-x[-2]))
        except:
            derivlast=Numeric.array([])
        result=Numeric.zeros(len(i1)+1,Numeric.Float)
        result[1:len(i1)]=0.5*((num[0:-1]/deno[0:-1])+\
                                     (num[1:]/deno[1:]))
        if len(derivfirst):
            result[0]=derivfirst
        else:
            result[0]=result[1]*1.0
        if len(derivlast):
            result[-1]=derivlast       
        else:
            result[-1]=result[-2]*1.0

        if type(ydata) == type(Numeric.array([])):
            return result
        else:
            return result.list

    def pretreat(self,xdata,ydata,xmin=None,xmax=None,):
        if xmax is None:
            xmax = max(xdata)
        if xmin is None:
            xmin = min(xdata)
        #sort data
        i1=Numeric.argsort(xdata)
        xdata=Numeric.take(xdata,i1)
        ydata=Numeric.take(ydata,i1)

        #take values between limits
        i1 =  Numeric.nonzero(xdata<=xmax)
        xdata = Numeric.take(xdata,i1)
        ydata = Numeric.take(ydata,i1)

        i1 =  Numeric.nonzero(xdata>=xmin)
        xdata = Numeric.take(xdata,i1)
        ydata = Numeric.take(ydata,i1)
        #OK with the pre-treatment
        return xdata,ydata

    def smooth(self,*vars,**kw):
        """
        smooth(self,*vars,**kw)
        Usage: self.smooth(y)
               self.smooth(y=y)
               self.smooth()
        """
        if 'y' in kw:
            ydata=kw['y']
        elif len(vars) > 0:
            ydata=vars[0]
        else:
            ydata=self.y
        f=[0.25,0.5,0.25]
        result=Numeric.array(ydata)
        if len(result) > 1:
            result[1:-1]=Numeric.convolve(result,f,mode=0)
            result[0]=0.5*(result[0]+result[1])
            result[-1]=0.5*(result[-1]+result[-2])
        if type(ydata) == type(Numeric.array([])):
            return result
        else:
            return result.list

    def squarefilter(self,*vars):
        if len(vars) > 0:
      	    y=vars[0]
        else:
       	    y=self.y
        if len(vars) > 1:
            width=vars[1]
        elif 'FwhmPoints' in self.fitconfig:
        	width=self.fitconfig['FwhmPoints']
        else:
         	width=5
        w = int(width) + ((int(width)+1) % 2)
        u = int(w/2)
        coef=Numeric.zeros((2*u+w),Numeric.Float)
        coef[0:u]=-0.5/float(u)
        coef[u:(u+w)]=1.0/float(w)
        coef[(u+w):len(coef)]=-0.5/float(u)
        if len(y) == 0:
            if type(y) == type([]):
        	    return []
            else:
         	    return Numeric.array([])
        else:
            if len(y) < len(coef):
          	    return y
            else:
                result=Numeric.zeros(len(y),Numeric.Float)
                result[(w-1):-(w-1)]=Numeric.convolve(y,coef,0)
                result[0:w-1]=result[w-1]
                result[-(w-1):]=result[-(w+1)]
                #import SimplePlot
                #SimplePlot.plot([self.xdata,y,result],yname='filter')
       	        return result
   		
       
def test():
    import SpecfitFunctions
    a=SpecfitFunctions.SpecfitFunctions()
    x = Numeric.arange(1000).astype(Numeric.Float)
    p1 = Numeric.array([1500,100.,50.0])
    p2 = Numeric.array([1500,700.,50.0])
    y = a.gauss(p1,x)+1
    y = y + a.gauss(p2,x)
    fit=Specfit()
    fit.setdata(x=x,y=y)
    fit.importfun("SpecfitFunctions.py")
    fit.settheory('Gaussians')
    #print fit.configure()
    fit.setbackground('Constant')
    if 1:
        fit.estimate()
        fit.startfit()
    else:
        fit.mcafit()
    print("Searched parameters = ",[1,1500,100.,50.0,1500,700.,50.0])
    print("Obtained parameters : ")
    for param in fit.paramlist:
        print(param['name'],' = ',param['fitresult'])
    print("chisq = ",fit.chisq)
    print("Attempting a plot")
    import PyQt4.Qt as qt
    import ScanWindow
    qtApp = qt.QApplication([])
    sw = ScanWindow.ScanWindow()
    sw.addCurve(x, y, legend="Data")
    yfit = fit.gendata()
    sw.addCurve(x, yfit, legend="Fit")
    sw.addCurve(x, y-yfit, legend="Data-Fit")
    sw.show()
    qtApp.exec_()


if __name__ == "__main__":
    test()
