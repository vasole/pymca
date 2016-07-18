from PyMca5 import Plugin1DBase
class Shifting(Plugin1DBase.Plugin1DBase):

    def getMethods(self, plottype=None):
        return ["Shift"]
    def getMethodToolTip(self, methodName):
        if methodName != "Shift":
            raise InvalidArgument("Method %s not valid" % methodName)
        return "Subtract minimum, normalize to maximum, and shift up by 0.1"

    def applyMethod(self, methodName):
        if methodName != "Shift":
            raise ValueError("Method %s not valid" % methodName)
        allCurves = self.getAllCurves()
        increment = 0.1
        for i in range(len(allCurves)):
            x, y, legend, info = allCurves[i][:4]
            delta = float(y.max() - y.min())
            if delta < 1.0e-15:
                delta = 1.0
            y = (y - y.min())/delta + i * increment
            if i == (len(allCurves) - 1):
                replot = True
            else:
                replot = False
            if i == 0:
                replace = True
            else:
                replace = False
            self.addCurve(x, y, legend=legend + " %.2f" % (i * increment),
                                info=info, replace=replace, replot=replot)
     
MENU_TEXT="Simple Vertical Shift"
def getPlugin1DInstance(plotWindow, **kw):
        ob = Shifting(plotWindow)
        return ob
