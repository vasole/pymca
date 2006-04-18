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
import Numeric
class SimpleMath:
    def derivate(self,xdata,ydata):
        f=[1,-1]
        x=Numeric.array(xdata)
        y=Numeric.array(ydata)
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
        xplot=Numeric.zeros(len(i1)+1,Numeric.Float)
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
        xplot[0]=x[0]
        xplot[-1]=x[-1]
        xplot[1:(len(i1)+1)]=Numeric.take(x,i1)
        return xplot,result
