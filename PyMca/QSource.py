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
import sys
if 'qt' in sys.modules:
    import qt
else:
    try:
        import PyQt4.Qt as qt
    except:
        import qt
QTVERSION = qt.qVersion()

SOURCE_EVENT = qt.QEvent.User

if QTVERSION < '4.0.0':
    class SourceEvent(qt.QCustomEvent):
        def __init__(self,dict=None):
            if dict is None: dict = {}
            qt.QCustomEvent.__init__(self, SOURCE_EVENT)
            self.dict = dict
else:
    class SourceEvent(qt.QEvent):
        def __init__(self,dict=None):
            if dict is None: dict = {}
            self.dict = dict
            qt.QEvent.__init__(self, SOURCE_EVENT)

import time
import thread
import weakref

class QSource(qt.QObject):
    def __init__(self):
        qt.QObject.__init__(self, None) #no parent

        self.surveyDict = {}
        self._pollTime = 0.5 #500 ms
        self.pollerThreadId = None
       

    def setPollTime(self, pollTime):
        """Set polling time (in milliseconds)"""
        self._pollTime = max(pollTime * 0.001, 0.001)

        return self._pollTime * 1000


    def getPollTime(self):
        return self._pollTime * 1000

    
    def addToPoller(self, dataObject):
        """Set polling for data object"""
        sourceName = dataObject.info['SourceName']

        if sourceName != self.sourceName:
            raise KeyError,"Trying to survey key %s on wrong source %s" % (self.sourceName,dataObject.info['SourceName'])             
      
        #that is general to any source        
        key        = dataObject.info['Key']

        def dataObjectDestroyed(ref, dataObjectKey=key):
            #print 'data object destroyed, key was %s' % key
            i = self.surveyDict[dataObjectKey].index(ref)
            """
            event = SourceEvent()
            event.dict={}
            event.dict['Key']   = key
            event.dict['event'] = 'destroyed'
            event.dict['id']    = self.surveyDict[key]
            qt.QApplication.postEvent(self.sourceObject, event)
            """
            del self.surveyDict[dataObjectKey][i]
            
            if len(self.surveyDict[dataObjectKey]) == 0:
                del self.surveyDict[dataObjectKey]

        # create a weak reference to the dataObject and we call it dataObjectRef
        dataObjectRef=weakref.proxy(dataObject, dataObjectDestroyed)
        
        try:
            if dataObjectRef not in self.surveyDict[key]:
                self.surveyDict[key].append(dataObjectRef)
        except KeyError:
            self.surveyDict[key] = [dataObjectRef]

        if self.pollerThreadId is None:
            # start a new polling thread
            #print "starting new thread"
            self.pollerThreadId = thread.start_new_thread(self.__run, ())


    def __run(self):
        #print "RUN"
        while len(self.surveyDict) > 0:
            #for key in self.surveyDict is dangerous
            # runtime error: dictionnary changed during iteration
            # a mutex is needed
            #print "IN LOOP"
            for key in self.surveyDict:
                #print " CUCU"
                if self.isUpdated(self.sourceName, key):
                    #print self.sourceName,key,"s updated"
                    if len(self.surveyDict[key]):
                        #there are still instances of dataObjects
                        event = SourceEvent()
                        event.dict['Key']   = key
                        event.dict['event'] = 'updated'
                        event.dict['id']    = self.surveyDict[key]

                        try:
                            qt.qApp.processEvents()
                            if QTVERSION < '4.0.0':qt.qApp.lock()
                            qt.QApplication.postEvent(self, event)
                        finally:
                            if QTVERSION < '4.0.0':qt.qApp.unlock()
            #print "sleeping"
            time.sleep(self._pollTime)
            #print " woke up"
            
        self.pollerThreadId = None
