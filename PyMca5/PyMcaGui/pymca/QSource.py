#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()
DEBUG = 0
SOURCE_EVENT = qt.QEvent.User

class SourceEvent(qt.QEvent):
    def __init__(self, ddict=None):
        if ddict is None:
            ddict = {}
        self.dict = ddict
        qt.QEvent.__init__(self, SOURCE_EVENT)

import time
try:
    import thread
except ImportError:
    import _thread as thread
import weakref

class QSource(qt.QObject):
    sigUpdated = qt.pyqtSignal(object)
    def __init__(self):
        qt.QObject.__init__(self, None) #no parent

        self.surveyDict = {}
        self.selections = {}
        self._pollTime = 0.7 #700 ms
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
            raise KeyError("Trying to survey key %s on wrong source %s" % (self.sourceName,dataObject.info['SourceName']))

        #that is general to any source
        key        = dataObject.info['Key']
        reference        = id(dataObject)

        def dataObjectDestroyed(ref, dataObjectKey=key, dataObjectRef=reference):
            if DEBUG:
                print('data object destroyed, key was %s' % dataObjectKey)
                print('data object destroyed, ref was 0x%x' % dataObjectRef)
                print("self.surveyDict[key] = ",self.surveyDict[key])

            n = len(self.surveyDict[dataObjectKey])
            if n > 0:
                n = list(range(n))
                n.reverse()
                for i in n:
                    if not len(dir(self.surveyDict[dataObjectKey][i])):
                        del self.surveyDict[dataObjectKey][i]

            if len(self.surveyDict[dataObjectKey]) == 0:
                del self.surveyDict[dataObjectKey]

            if DEBUG:
                print("SURVEY DICT AFTER DELETION = ", self.surveyDict)
            return

        # create a weak reference to the dataObject and we call it dataObjectRef
        dataObjectRef=weakref.proxy(dataObject, dataObjectDestroyed)

        try:
            if dataObjectRef not in self.surveyDict[key]:
                self.surveyDict[key].append(dataObjectRef)
                self.selections[key].append((id(dataObjectRef), dataObjectRef.info))
        except KeyError:
            self.surveyDict[key] = [dataObjectRef]
            self.selections[key] = [(id(dataObjectRef), dataObjectRef.info)]
        except ReferenceError:
            if DEBUG:
                print("NOT ADDED TO THE POLL dataObject = ", dataObject)
            return

        if DEBUG:
            print("SURVEY DICT AFTER ADDITION = ", self.surveyDict)

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
            if DEBUG:
                print("In loop")
            dummy = list(self.surveyDict.keys())
            eventsToPost = {}
            #for key in self.surveyDict:
            for key in dummy:
                if key not in eventsToPost: 
                    eventsToPost[key] = []
                if self.isUpdated(self.sourceName, key):
                    if DEBUG:
                        print(self.sourceName,key,"is updated")
                    try:
                        if len(self.surveyDict[key]):
                            #there are still instances of dataObjects
                            event = SourceEvent()
                            event.dict['Key']   = key
                            event.dict['event'] = 'updated'
                            event.dict['id']    = self.surveyDict[key]
                            scanselection = False
                            if 'scanselection' in self.surveyDict[key][0].info:
                                scanselection = \
                                  self.surveyDict[key][0].info['scanselection']
                            if (key == 'SCAN_D') or scanselection:
                                event.dict['scanselection'] = True
                            else:
                                event.dict['scanselection'] = False
                            eventsToPost[key].append(event)
                        else:
                            del self.surveyDict[key]
                            del self.selections[key]
                    except KeyError:
                        if DEBUG:
                            print("key error in loop")
                        pass
            for key in eventsToPost:
                for event in eventsToPost[key]:
                    qt.QApplication.postEvent(self, event)
            qt.QApplication.instance().processEvents()
            time.sleep(self._pollTime)
            if DEBUG:
                print("woke up")

        self.pollerThreadId = None
        self.selections = {}

