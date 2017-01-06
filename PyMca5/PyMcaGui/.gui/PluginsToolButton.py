
# from silx.gui.plot.PlotActions import PlotAction
from silx.gui import qt

from PyMca5.PyMcaGraph.PluginLoader import PluginLoader
from PyMca5.PyMcaGui.plotting.PyMca_Icons import IconDict


class PluginsToolButton(qt.QToolButton, PluginLoader):
    """PlotAction providing a context menu loaded with PyMca plugins

    :param plot: reference to related plot widget
    :param parent: Parent QWidget widget
    """
    def __init__(self, plot, parent=None):

        qt.QToolButton.__init__(self, parent)
        self.setIcon(qt.QIcon(qt.QPixmap(IconDict["plugin"])))
        self.setToolTip("Call/Load 1D Plugins")

        # fill attr pluginList and pluginInstanceDict with existing plugins
        PluginLoader.__init__(self, parent)

        self.plot = plot




        # proxies to methods needed by plugins
        self.getActiveCurve = self.plot.getActiveCurve
        self.getAllCurves = self.plot.getAllCurves
        self.addCurve = self.plot.addCurve
        self.getGraphXLimits = self.plot.getGraphXLimits
        # TODO : all methods



