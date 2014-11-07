# /*#########################################################################
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
# ###########################################################################*/
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module provides an implementation of state machines to support interaction
"""


# state machine ###############################################################


class State(object):
    """Base class for states of a state machine"""
    def __init__(self, machine):
        """
        :param machine: The state machine instance this state belongs to
        :type machine: StateMachine
        """
        self.machine = machine

    def goto(self, state, *args, **kwargs):
        """Performs a transition to state
        Extra arguments are passed to the enter method of state
        :param State state: The class of the state to go to
        """
        self.machine._goto(state, *args, **kwargs)

    def enter(self, *args, **kwargs):
        """Called when the state machine enters this state"""
        pass


class StateMachine(object):
    """State machine controller"""
    def __init__(self, initState, *args, **kwargs):
        """Create a state machine controller with an initial state
        Extra arguments are passed to the enter method of the initState
        :param State initState: Class of the initial state of the state machine
        """
        self._goto(initState, *args, **kwargs)

    def _goto(self, state, *args, **kwargs):
        self.state = state(self)
        self.state.enter(*args, **kwargs)

    def handleEvent(self, eventName, *args, **kwargs):
        """Process an event with the state machine
        :param str eventName: Name of the event to handle
        """
        handlerName = 'on' + eventName[0].upper() + eventName[1:]
        try:
            handler = getattr(self.state, handlerName)
        except AttributeError:
            try:
                handler = getattr(self, handlerName)
            except AttributeError:
                handler = None
        if handler is not None:
            handler(*args, **kwargs)


# clicOrDrag ##################################################################

LEFT_BTN, RIGHT_BTN, MIDDLE_BTN = 'left', 'right', 'middle'


class ClicOrDrag(StateMachine):
    """State machine for left and right clic and left drag interaction"""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto(ClicOrDrag.ClicOrDrag, x, y)
            elif btn == RIGHT_BTN:
                self.goto(ClicOrDrag.RightClic, x, y)

    class RightClic(State):
        def onMove(self, x, y):
            self.goto(ClicOrDrag.Idle)

        def onRelease(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.clic(x, y, btn)
                self.goto(ClicOrDrag.Idle)

    class ClicOrDrag(State):
        def enter(self, x, y):
            self.initPos = x, y

        def onMove(self, x, y):
            self.goto(ClicOrDrag.Drag, self.initPos, (x, y))

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.clic(x, y, btn)
                self.goto(ClicOrDrag.Idle)

    class Drag(State):
        def enter(self, initPos, curPos):
            self.initPos = initPos
            self.machine.beginDrag(*initPos)
            self.machine.drag(*curPos)

        def onMove(self, x, y):
            self.machine.drag(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endDrag(self.initPos, (x, y))
                self.goto(ClicOrDrag.Idle)

    def __init__(self):
        super(ClicOrDrag, self).__init__(self.Idle)

    def clic(self, x, y, btn):
        pass

    def beginDrag(self, x, y):
        pass

    def drag(self, x, y):
        pass

    def endDrag(self, x, y):
        pass


# main ########################################################################

if __name__ == "__main__":
    class DumpClicOrDrag(ClicOrDrag):
        def clic(self, x, y, btn):
            print('clic', x, y, btn)

        def beginDrag(self, x, y):
            print('beginDrag', x, y)

        def drag(self, x, y):
            print('drag', x, y)

        def endDrag(self, x, y):
            print('endDrag', x, y)

    clicOrDrag = DumpClicOrDrag()
    for event in (('press', 10, 10, LEFT_BTN),
                  ('release', 10, 10, LEFT_BTN),
                  ('press', 10, 10, LEFT_BTN),
                  ('move', 11, 10),
                  ('move', 12, 10),
                  ('release', 12, 10, LEFT_BTN)):
        print('Event:', event)
        clicOrDrag.handleEvent(*event)
