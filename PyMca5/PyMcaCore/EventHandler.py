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
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""
This module implements an event handler class.

This a communication system between objects based on a pattern observer /
producer. The producer generates events, and the observer are listening
to events. The event handler is designed to manage this communication.

The communication is completely synchronous. (This is differnet to the X
mainloop where callbacks are not interrupted.)

The events have a hierarchy which is specified by a fullname just like other
python classes (i.e The callback of the NewDataEvent.XNewDataEvent event is
called when its parent the NewDataEvent is fired and also when the
XNewDataEvent is fired).

The pattern should be the following:

Class1 sends events. It will be responsible to create them in the
beginning with:
   myevent = eh.create("myevent")
and later send the event with
   eh.event (myevent, arg1, arg2, ...)

Class2 receives events. It will be responsible to register its interest in
the events and specify the methods to be called

The Main program should create the eventhandler
   eh = Eventhandler()
and pass it to the constructor of the classes
   Class1(eh=eh)
   Class2(eh=eh)

Some conventions:

  * All the registered events should have public documented methods as
    alternatives. The classes should not rely on the fact that an eventhandler
    is passed
  * The eventhandler argument should be a keyword called eh and defaults to
    None and should not be necessary. You should forsee that the functionality
    can be used via std callbacks (i.e in the constructor selectcb=select),
    overriding std methods (i.e. select() or simple methods in your class to
    set a callback or provide the information directly (i.e SetSelectCB(),
    GetSelection)

Events: Classes derived from the Event class
Full event names: A string with the event name fully specified (i.e. a.b.c)

"""
__version__ = '0.1Beta'

import logging

_logger = logging.getLogger(__name__)


class Event(object):
  pass

class OneEvent(object):
  def __init__(self, parent = None, event = None):
    self.parent = parent
    self.event = event
    self.callbacks = []
    self.creator = None
    self.created = 0

class EventHandler(object):
    def __init__(self):
      self.callbacks = {}
      self.fulldict = {}
      self.rootevent = OneEvent(event = Event)
      self.events = {}

    def _create(self, fulleventstr, myid = None):
      try:
        return self.fulldict[fulleventstr]
      except KeyError:
        try:
          idx = fulleventstr.rindex(".") + 1
          parentstr = fulleventstr[:idx-1]
          parent = self._create(parentstr)
        except ValueError:
          parent = self.rootevent
          idx = 0
      #event = new.classobj(fulleventstr[idx:], (parent.event,), globals())
      event = type(fulleventstr[idx:], (parent.event,), globals())
      self.fulldict[fulleventstr] = OneEvent(event = event, parent = parent)
      return self.fulldict[fulleventstr]

    def create(self, fulleventstr, myid = None):
      """ Create the event. This call will take a full classname a.b.c and
          create the event calls and all the parent classes if necessary. It
          returns the eventclassobject which can be used to fire the event
          later. It is no error to create the class after registering for
          it, but it is an error to fire an event before creating it. Normally
          the event producer is responsible for creating it
      """
      oe = self._create(fulleventstr, myid = myid)
      oe.creator = myid
      oe.created = 1
      self.preparefastevents()
      return oe.event

    def register(self, fulleventstr, callback, myid = None, source = None):
      """ Register the event a.b.c with callback . You have to specify the
          full name of the event class as it might be created during this call.
          A later create call with the same event will just confirm this
          creation. You can specify an id for yourself and an id for the
          source you would like to listen to. The source restrictions are not
          yet implemented because of performance considerations.
      """
      oe = self._create(fulleventstr)
      oe.callbacks.append((callback, myid, source))
      self.preparefastevents()
      return oe.event

    def unregister(self, fulleventstr, callback, myid = None):
      """ Unregister the callback from the eventclass a.b.c. The id has to
          be specified if it has been specified on registering the callback
      """
      try:
        oe = self.fulldict[fulleventstr]
        for cb, regid, source in oe.callbacks:
          if cb == callback and regid == myid:
            oe.callbacks.remove((cb, myid, source))
      except KeyError:                      # there is no such event
        pass
      self.preparefastevents()

    def dumptostr(self, fullname, cbflag = 1):
      s = "%s: " % fullname
      try:
        oe = self.fulldict[fullname]
      except KeyError:
        return  s + "undefined\n"

      if oe.created == 0:
        creator = "<ONLY REGISTERED>"
      elif oe.creator is None:
        creator = "creator not specified"
      else:
        creator = "created by " + oe.creator

      s = s + "(%s)\n" % creator

      if cbflag:
        for cb, regid, source in oe.callbacks:
          try:
            cbname = cb.__name__
          except AttributeError:
            cbname = "%s" % cb

          s = s + "   %s" % cbname
          if regid:
            s = s + " (reg by: %s)" % str(regid)
          if source:
            s = s + " (only: %s)" % str(source)
          s = s + "\n"

      return s

    def dumpalltostr(self):
      s = ""
      for fullname in self.fulldict.keys():
        s = s + self.dumptostr(fullname)
      return s

    def preparefastevents(self):
      """ calculate the callback functions for all possible events """
      self.events = {}
      self.callbacks = {}
      for fullev, oe in self.fulldict.items():
        events = fullev.split(".")
        self.events[events[-1]] = oe.event  # only for our callers
        cbs = [x[0] for x in self.fulldict[fullev].callbacks]
        for i in range(len(events)):
          evname = ".".join(events[:i+1])
          ev = self.fulldict[evname].event
          try:
            self.callbacks[ev] = self.callbacks[ev] + cbs
          except:
            self.callbacks[ev] = cbs

    def event(self, event, *args, **kw):
      """ Fire the event with arguments and keywords """
      if event in self.callbacks.keys():
        for cb in self.callbacks[event]:
            cb(*args, **kw)
      else:
        _logger.warning("Warning: missing event: %s", event)

    def getfullevents(self):
      """ return a list with fully specified event names (a.b.c) """
      return self.fulldict.keys()

    def getevents(self):
      """ return a list with name item tuples. """
      evdict = {}
      for fullev in self.fulldict.keys():
        events = fullev.split(".")
        dict = evdict
        for i in range(len(events)):
          evname = events[i]
          if not (evname in dict):
            dict[evname] = {}
          dict = dict[evname]
      return self._dict2tup(evdict)

    def _dict2tup(self, dict):
      li = []
      for key, item in dict.items():
        if item == {}:
          li.append((key, 0))
        else:
          li.append((key, self._dict2tup(item)))
      return li

def test(eh = None):
    """EventHandler class test function"""
    def callback1(data, more=None):
         print('Hi callback 1 (Data) with data : %s and %s' % (data, more))

    def callback2(data, more=None):
         print('Hi callback 2 (XData) with data : %s and %s' % (data, more))

    def callback3(data, more=None):
         print('Hi callback 3 (Ydata) with data : %s and %s' % (data, more))

    if eh is None:
      eh = EventHandler()

    NewDataEvent = eh.create("NewDataEvent")
    XNewDataEvent = eh.create("NewDataEvent.XNewDataEvent")
    YNewDataEvent = eh.create("NewDataEvent.YNewDataEvent")
    eh.register("NewDataEvent", callback1)
    eh.register("NewDataEvent.XNewDataEvent" , callback2)
    eh.register("NewDataEvent.YNewDataEvent" , callback3)
    print("%s" % eh.getevents())

    eh.event(XNewDataEvent, "this is data for 2")
    eh.event(NewDataEvent, "this is data for 1,2,3", more=[1,2,3])
    eh.event(eh.events["YNewDataEvent"], "more for 3")

    try:
        eh.event("XNewDataEvent", "this is data again")
    except KeyError:
        print("Error: String as Event has been detected sucessfully")

    eh.unregister("NewDataEvent", callback1)
    eh.event(NewDataEvent, "this is data again again")

if __name__ == '__main__':
    test()
