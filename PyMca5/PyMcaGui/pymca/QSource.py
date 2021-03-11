#/*##########################################################################
# Copyright (C) 2004-2021 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import sys
import logging
import time
import weakref
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()
_logger = logging.getLogger(__name__)
SOURCE_EVENT = qt.QEvent.registerEventType()

try:
    import thread
except ImportError:
    import _thread as thread

class SourceEvent(qt.QEvent):
    def __init__(self, ddict=None):
        if ddict is None:
            ddict = {}
        self.dict = ddict
        qt.QEvent.__init__(self, SOURCE_EVENT)

class QSource(qt.QObject):
    sigUpdated = qt.pyqtSignal(object)
    def __init__(self):
        qt.QObject.__init__(self, None) #no parent

        self.surveyDict = {}
        self.selections = {}
        self.setPollTime(0.7) # 700 milliseconds
        self.pollerThreadId = None

    def setPollTime(self, pollTime):
        """Set polling time (in seconds)"""
        self._pollTime = max(pollTime, 0.1)
        return self._pollTime

    def getPollTime(self):
        return self._pollTime

    def addToPoller(self, dataObject):
        """Set polling for data object"""
        sourceName = dataObject.info['SourceName']

        if sourceName != self.sourceName:
            raise KeyError("Trying to survey key %s on wrong source %s" % (self.sourceName,dataObject.info['SourceName']))

        #that is general to any source
        key        = dataObject.info['Key']
        reference        = id(dataObject)

        def dataObjectDestroyed(ref, dataObjectKey=key, dataObjectRef=reference):
            _logger.debug('data object destroyed, key was %s', dataObjectKey)
            _logger.debug('data object destroyed, ref was 0x%x', dataObjectRef)
            _logger.debug("self.surveyDict[key] = %s", self.surveyDict[key])

            n = len(self.surveyDict[dataObjectKey])
            if n > 0:
                ns = list(range(n))
                newlist = []
                for i in ns:
                    try:
                        if len(dir(self.surveyDict[dataObjectKey][i])):
                            newlist.append(self.surveyDict[dataObjectKey][i])
                    except ReferenceError:
                        pass

                self.surveyDict[dataObjectKey] = newlist

            if len(self.surveyDict[dataObjectKey]) == 0:
                del self.surveyDict[dataObjectKey]

            _logger.debug("SURVEY DICT AFTER DELETION = %s", self.surveyDict)
            return

        # create a weak reference to the dataObject and we call it dataObjectRef
        dataObjectRef=weakref.proxy(dataObject, dataObjectDestroyed)

        try:
            _logger.debug("Dealing with data object reference %s", dataObjectRef)
            if key not in self.surveyDict:
                self.surveyDict[key] = [dataObjectRef]
                self.selections[key] = [(id(dataObjectRef), dataObjectRef.info)]
            elif dataObjectRef not in self.surveyDict[key]:
                _logger.debug("dataObject reference ADDED")

                self.surveyDict[key].append(dataObjectRef)
                self.selections[key].append((id(dataObjectRef), dataObjectRef.info))
            else:
                _logger.debug("dataObject reference IGNORED")
        except KeyError:
            print("ADDING BECAUSE OF KEY ERROR")
            self.surveyDict[key] = [dataObjectRef]
            self.selections[key] = [(id(dataObjectRef), dataObjectRef.info)]
        except ReferenceError:
            _logger.debug("NOT ADDED TO THE POLL dataObject = %s", dataObject)
            return

        if self.pollerThreadId is None:
            # start a new polling thread
            _logger.debug("starting new thread")
            self.pollerThreadId = thread.start_new_thread(self.__run, ())

    def __run(self):
        _logger.debug("In QSource __run method")
        while len(self.surveyDict) > 0:
            #for key in self.surveyDict is dangerous
            # runtime error: dictionary changed during iteration
            # a mutex is needed
            _logger.debug("In loop")
            dummy = list(self.surveyDict.keys())
            eventsToPost = {}
            #for key in self.surveyDict:
            for key in dummy:
                if key not in eventsToPost: 
                    eventsToPost[key] = []
                if self.isUpdated(self.sourceName, key):
                    _logger.debug("%s %s is updated", self.sourceName, key)
                    try:
                        if len(self.surveyDict[key]):
                            #there are still instances of dataObjects
                            event = SourceEvent()
                            event.dict['Key']   = key
                            event.dict['event'] = 'updated'
                            event.dict['id']    = self.surveyDict[key]
                            scanselection = False
                            info = self.surveyDict[key][0].info
                            if "scanselection" in info:
                                scanselection = info['scanselection']
                            elif "selectiontype" in info:
                                _logger.debug("selectiontype %s", info["selectiontype"])
                                if info["selectiontype"] == "1D":
                                    scanselection = True
                            if (key == 'SCAN_D') or scanselection:
                                event.dict['scanselection'] = True
                            else:
                                event.dict['scanselection'] = False
                            eventsToPost[key].append(event)
                        else:
                            del self.surveyDict[key]
                            del self.selections[key]
                    except:
                        _logger.debug("error in loop %s", sys.exc_info())
                        del self.surveyDict[key]
                        del self.selections[key]
                        pass
            for key in eventsToPost:
                for event in eventsToPost[key]:
                    qt.QApplication.postEvent(self, event)
            qt.QApplication.instance().processEvents()
            time.sleep(self._pollTime)
            _logger.debug("woke up")

        self.pollerThreadId = None
        self.selections = {}
