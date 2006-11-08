import sys
from QtBlissGraph import qt
if __name__ == "__main__":
    app = qt.QApplication([])
import QtBlissGraph
from Icons import IconDict
import Numeric
import DataObject
import copy

QTVERSION = qt.qVersion()
DEBUG = 0

MATPLOTLIB = 0
#QtBlissgraph should have the same colors
colordict = {}
colordict['blue']  = '#0000ff'
colordict['red']   = '#ff0000'
colordict['green'] = '#00ff00'
colordict['black'] = '#000000'
colordict['white'] = '#ffffff'
colordict['pink'] = '#ff66ff'
colordict['brown'] = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['yellow'] = '#ffff00'
colordict['darkgreen']  = 'g'
colordict['darkbrown'] = '#660000' 
colordict['magenta'] = 'm' 
colordict['cyan']  = 'c'
colordict['bluegreen']  = '#33ffff'
colorlist  = [colordict['black'],
              colordict['red'],
              colordict['blue'],
              colordict['green'],
              colordict['pink'],
              colordict['brown'],
              colordict['cyan'],
              colordict['orange'],
              colordict['violet'],
              colordict['bluegreen'],
              colordict['grey'],
              colordict['magenta'],
              colordict['darkgreen'],
              colordict['darkbrown'],
              colordict['yellow']]
#qt is not necessary, but I use it in the rest of the application
#without QT:
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#
try:
    from matplotlib import rcParams
    #matplotlib.rcParams['numerix'] = "numeric"
    rcParams['numerix'] = "numeric"
    from matplotlib.font_manager import FontProperties
    #2D stuff
    if QTVERSION < '4.0.0':
        from matplotlib.backends.backend_qtagg import FigureCanvasQT, FigureCanvasQTAgg
    else:
        from matplotlib.backends.backend_qt4agg import  FigureCanvasQT, FigureCanvasQTAgg
    #3D stuff
    from matplotlib.figure import Figure
    import matplotlib.axes3d as axes3d
    MATPLOTLIB = 1
    class FigureCanvas(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=None, height=None, dpi=None,
                     xlist=None, ylist = None,
                     dataobject = None, fixed_print_size = False, toolbar = False,
                     **kw):
            if (width == None) or (height == None):
                if dpi is not None:
                    self.fig = Figure(dpi=dpi)
                else:
                    self.fig = Figure()
            FigureCanvasQTAgg.__init__(self, self.fig)
            self.p = None
            if toolbar:
                if parent is None:
                    self.p = qt.QWidget()
                    layout = qt.QVBoxLayout(self.p)
                layout = self.p.layout()
                layout.setMargin(0)
                layout.setSpacing(0)
                button1 = qt.QPushButton(parent)
                button1.setText("Save")
                layout.addWidget(button1)
                #layout.addWidget(HorizontalSpacer(self.p))
                #button2 = qt.QPushButton(parent)
                #button2.setText("Print")
                #layout.addWidget(button2)
            if QTVERSION < '4.0.0':
                if parent is not None:
                    self.reparent(parent, QPoint(0, 0))
                else:
                    #self.reparent(parent, QPoint(0, 0))
                    layout.addWidget(self)
            else:
                if parent is not None:
                    self.setParent(parent)
                    self.move(QPoint(0,0))
                else:
                    if toolbar:
                        layout.addWidget(self)
            self._fixed_print_size = fixed_print_size
            if dataobject is not None:
                self.plotDataObject(dataobject)
            else:
                self.dataObject = None

            if xlist is not None:
                if ylist is not None:
                    self.plotXYList(xlist, ylist, **kw)

            if toolbar:self.p.show()

        def plotDataObject(self, dataObject, **kw):
            if len(dataObject.x) == 1: self._plot2DDataObject(dataObject, **kw)
            else: self._plot3DDataObject(dataObject, **kw)

        def plotXYList(self, xlist, ylist, **kw):
            logy    = False
            legends = True
            legendslist = []
            loc = (1.01, 0.0)
            labelsep = 0.02
            xlabel = None
            ylabel = None
            for key in kw:
                if key.upper() == "LOC":
                    loc = kw[key]
                    continue
                if key.upper() == "LOGY":
                    if kw[key]:logy = True
                    else:logy = False
                    continue
                if key.upper() == "XLABEL":
                    xlabel = kw[key]
                    continue
                if key.upper() == "YLABEL":
                    ylabel = kw[key]
                    continue
                if key.upper() == "LEGENDS":
                    if kw[key]:legends = True
                    else:      legends = False
                    continue
                if key.upper() == "LEGENDSLIST":
                    legendslist = kw[key]
                    continue
                if key.upper() == "XMIN":
                    if kw[key]:xmin = kw[key]
                    else:      xmin = xdata[0]
                    continue
                if key.upper() == "XMAX":
                    if kw[key]:xmax = kw[key]
                    else:      xmax = xdata[-1]
                    continue
                if key.upper() == "YMIN":
                    if kw[key]:ymin = kw[key]
                    else:      ymin = None
                    continue
                if key.upper() == "YMAX":
                    if kw[key]:ymax = kw[key]
                    else:      ymax = None

            icolor = 0
            if len(legendslist) == 0:legends = False
            if not legends:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .85, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .85, .8])
            else:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .7, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .7, .8])
            if logy:
                axfunction = ax.semilogy
            else:
                axfunction = ax.plot
            ax.set_axisbelow(True)
            i = 0
            lshape = "-"
            for iplot in range(len(xlist)):
                xplot = xlist[iplot]
                yplot = ylist[iplot]
                axfunction(xplot, yplot, ls=lshape, color=colorlist[i],lw=1.5)
                i += 1
                if i == len(colorlist):
                    i = 2   #black and red used only once
                    lshape = ".."

            if legends and len(legendslist):
                if len(legendslist) > 14:
                    #loc = (1.01, 0.0)  # outside at plot bottom
                    fontsize = 8
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                else:
                    fontsize = 10
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                    
                legend = ax.legend(legendslist,
                                   loc = loc,
                                   #fontname = "Times",
                                   prop = fontproperties,
                                   labelsep = labelsep,
                                   pad = 0.15)
                legend.draw_frame(drawframe)
            else:
                fontsize = 10
                fontproperties = FontProperties(size=fontsize)
            #if xlabel is not None:ax.set_xlabel(xlabel) #, prop=fontproperties)
            #if ylabel is not None:ax.set_ylabel(ylabel) #, prop=fontproperties)
            ax.set_xlim(xmin, xmax)
            self.fig.canvas.show()
            self.print_figure("test"+".png", dpi=300) #print quality independent
            #canvas.print_figure(filename+".eps", dpi=300) #of screen quality


        def _plot2DDataObject(self, dataObject, **kw):
            if self.p is not None:
                if QTVERSION < '4.0.0':
                    self.p.setCaption(dataObject.info['legend'])
                else:
                    self.p.setWindowTitle(dataObject.info['legend'])
            xdata = dataObject.x[0]
            ydata = dataObject.y[0]
            index = None
            if dataObject.m is None:
                mdata = [Numeric.ones(len(zdata)).astype(Numeric.Float)]
            elif len(dataObject.m[0]) > 0:
                if len(dataObject.m[0]) == len(ydata):
                    index = Numeric.nonzero(dataObject.m[0])
                    if not len(index): return
                    xdata = Numeric.take(xdata, index)
                    mdata = Numeric.take(dataObject.m[0], index)
                else:
                    raise "ValueError", "Monitor data length different than counter data"

            logy    = False
            legends = True
            legendslist = None
            loc = (1.01, 0.0)
            labelsep = 0.02
            for key in kw:
                if key.upper() == "LOC":
                    loc = kw[key]
                    continue
                if key.upper() == "LOGY":
                    if kw[key]:logy = True
                    else:logy = False
                    continue
                if key.upper() == "LEGENDS":
                    if kw[key]:legends = True
                    else:      legends = False
                    continue
                if key.upper() == "LEGENDSLIST":
                    legendslist = kw[key]
                    continue
                if key.upper() == "XMIN":
                    if kw[key]:xmin = kw[key]
                    else:      xmin = xdata[0]
                    continue
                if key.upper() == "XMAX":
                    if kw[key]:xmax = kw[key]
                    else:      xmax = xdata[-1]
                    continue
                if key.upper() == "YMIN":
                    if kw[key]:ymin = kw[key]
                    else:      ymin = None
                    continue
                if key.upper() == "YMAX":
                    if kw[key]:ymax = kw[key]
                    else:      ymax = None

            xplot = Numeric.take(xdata, index)
            mplot = Numeric.take(mdata, index)
            icolor = 0
            xlabel = 'X'
            ylabel = 'Y'
            if legendslist is None:
                legendslist = []
                sel = {}
                sel['selection'] = dataObject.info['selection']
                if sel['selection'] is not None:
                    if type(sel['selection']) == type({}):
                        if xlabel is None:
                            if sel['selection'].has_key('x'):
                                ilabel = sel['selection']['x'][0]
                                xlabel = dataObject.info['LabelNames'][ilabel]
                        if ylabel is None:
                            if sel['selection'].has_key('y'):
                                ilabel = sel['selection']['y'][0]
                                ylabel = dataObject.info['LabelNames'][ilabel]
                        if sel['selection'].has_key('y'):
                            for ilabel in  sel['selection']['y']:
                                legendslist.append(dataObject.info['LabelNames'][ilabel])

                else:legends = False
                
            if not legends:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .85, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .85, .8])
            else:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .7, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .7, .8])
            if logy:
                axfunction = ax.semilogy
            else:
                axfunction = ax.plot
            ax.set_axisbelow(True)

            self.dataObject = dataObject
            


            i = 0
            lshape = "-"
            for y in dataObject.y:
                yplot = Numeric.take(y, index)/mplot
                axfunction(xplot, yplot, ls=lshape, color=colorlist[i],lw=1.5)
                i += 1
                if i == len(colorlist):
                    i = 2   #black and red used only once
                    lshape = ".."

            if legends and len(legendslist):
                if len(legendslist > 14):
                    #loc = (1.01, 0.0)  # outside at plot bottom
                    fontsize = 8
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                else:
                    fontsize = 10
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                    
                legend = ax.legend(legendslist,
                                   loc = loc,
                                   fontname = "Times",
                                   prop = fontproperties,
                                   labelsep = labelsep,
                                   pad = 0.15)
                legend.draw_frame(drawframe)
            else:
                fontsize = 10
                fontproperties = FontProperties(size=fontsize)
            if xlabel is not None:ax.set_xlabel(xlabel, prop=fontproperties)
            if ylabel is not None:ax.set_ylabel(ylabel, prop=fontproperties)
            ax.set_xlim(xmin, xmax)
            self.fig.canvas.show()
            canvas.print_figure("test"+".png", dpi=300) #print quality independent
            #canvas.print_figure(filename+".eps", dpi=300) #of screen quality
        
        def _plot3DDataObject(self, dataObject, **kw):
            if self.p is not None:
                if QTVERSION < '4.0.0':
                    self.p.setCaption(dataObject.info['legend'])
                else:
                    self.p.setWindowTitle(dataObject.info['legend'])
            #let's try to generate a plot
            xdata = dataObject.x[0]
            ydata = dataObject.x[1]
            zdata = dataObject.y[0]
            index = None
            if dataObject.m is None:
                mdata = [Numeric.ones(len(zdata)).astype(Numeric.Float)]
            elif len(dataObject.m[0]) > 0:
                if len(dataObject.m[0]) == len(zdata):
                    index = Numeric.nonzero(dataObject.m[0])
                    if not len(index): return
                    xdata = Numeric.take(xdata, index)
                    ydata = Numeric.take(ydata, index)
                    zdata = Numeric.take(zdata, index)
                    mdata = Numeric.take(dataObject.m[0], index)
                else:
                    raise "ValueError", "Monitor data length different than counter data"
            zdata = zdata / mdata
            #self.fig = Figure(figsize=(6,3), dpi = 150)
            #self.fig = Figure()
            #self.fig.canvas = FigureCanvas(self.fig)
            #layout  = qt.QVBoxLayout(self.fig.canvas)
            #toolBar = qt.QWidget(self.fig.canvas)
            #toolBarLayout = qt.QHBoxLayout(toolBar)
            #tb      = qt.QToolButton(toolBar)
            #tb.setIconSet(self.savIcon)
            #self.connect(tb,qt.SIGNAL('clicked()'),self.printps)
            #qt.QToolTip.add(tb,'Prints the Graph')
            #self.toolBarLayout.addWidget(tb)
            #layout.addWidget(toolBar)
            ax = axes3d.Axes3D(self.fig)
            #f = ax.scatter3D
            f = ax.plot3D
            #f = ax.contourf
            for z in dataObject.y:
                if index is None:
                    zdata = z
                else:
                    zdata = Numeric.take(z, index)
                f(Numeric.ravel(xdata),
                         Numeric.ravel(ydata),
                         Numeric.ravel(zdata))
            ax.toolbar = MatplotlibToolbar()
            xlabel = 'X'
            ylabel = 'Y'
            zlabel = 'Z'
            sel = {}
            sel['selection'] = dataObject.info['selection']
            if sel['selection'] is not None:
                if type(sel['selection']) == type({}):
                    if sel['selection'].has_key('x'):
                        #proper scan selection
                        ilabel = sel['selection']['x'][0]
                        xlabel = dataObject.info['LabelNames'][ilabel]
                        ilabel = sel['selection']['x'][1]
                        ylabel = dataObject.info['LabelNames'][ilabel]
                        ilabel = sel['selection']['y'][0]
                        zlabel = dataObject.info['LabelNames'][ilabel]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            self.fig.canvas.show()
            self.dataObject = dataObject

        def resizeEvent( self, e ):
            FigureCanvasQT.resizeEvent( self, e )
            w = e.size().width()
            h = e.size().height()
            if DEBUG: print "FigureCanvasQtAgg.resizeEvent(", w, ",", h, ")"
            dpival = self.figure.dpi.get()
            if not self._fixed_print_size:
                winch = w/dpival
                hinch = h/dpival
                #self.figure.set_figsize_inches( winch, hinch )
                self.figure.set_size_inches( winch, hinch )
            self.draw()

        def sizeHint(self):
            w, h = self.get_width_height()
            return qt.QSize(w, h)

        def motion_notify_event(self, *var):
            print "catching"
            print var
except:
    if DEBUG:print "matplotlib not loaded"



class ScanWindow(qt.QWidget):
    def __init__(self, parent=None, name="Scan Window"):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setCaption(name)
        else:
            self.setWindowTitle(name)
        self._initIcons()
        self._build()
        self.fig = None
        self.graph.canvas().setMouseTracking(1)

        if QTVERSION < '4.0.0':
            self.connect(self.graph,
                         qt.PYSIGNAL("QtBlissGraphSignal"),
                         self._graphSignalReceived)
        else:
            self.connect(self.graph,
                         qt.SIGNAL("QtBlissGraphSignal"),
                         self._graphSignalReceived)
        self.dataObjectsDict = {}
        self.dataObjectsList = []

    def _initIcons(self):
        if QTVERSION < '4.0.0':
            self.normalIcon	= qt.QIconSet(qt.QPixmap(IconDict["normal"]))
            self.zoomIcon	= qt.QIconSet(qt.QPixmap(IconDict["zoom"]))
            self.roiIcon	= qt.QIconSet(qt.QPixmap(IconDict["roi"]))
            self.peakIcon	= qt.QIconSet(qt.QPixmap(IconDict["peak"]))

            self.zoomResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["zoomreset"]))
            self.roiResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["roireset"]))
            self.peakResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["peakreset"]))
            self.refreshIcon	= qt.QIconSet(qt.QPixmap(IconDict["reload"]))

            self.logxIcon	= qt.QIconSet(qt.QPixmap(IconDict["logx"]))
            self.logyIcon	= qt.QIconSet(qt.QPixmap(IconDict["logy"]))
            self.xAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["xauto"]))
            self.yAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["yauto"]))
            self.fitIcon	= qt.QIconSet(qt.QPixmap(IconDict["fit"]))
            self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))
            self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
            self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))
        else:
            self.normalIcon	= qt.QIcon(qt.QPixmap(IconDict["normal"]))
            self.zoomIcon	= qt.QIcon(qt.QPixmap(IconDict["zoom"]))
            self.roiIcon	= qt.QIcon(qt.QPixmap(IconDict["roi"]))
            self.peakIcon	= qt.QIcon(qt.QPixmap(IconDict["peak"]))

            self.zoomResetIcon	= qt.QIcon(qt.QPixmap(IconDict["zoomreset"]))
            self.roiResetIcon	= qt.QIcon(qt.QPixmap(IconDict["roireset"]))
            self.peakResetIcon	= qt.QIcon(qt.QPixmap(IconDict["peakreset"]))
            self.refreshIcon	= qt.QIcon(qt.QPixmap(IconDict["reload"]))

            self.logxIcon	= qt.QIcon(qt.QPixmap(IconDict["logx"]))
            self.logyIcon	= qt.QIcon(qt.QPixmap(IconDict["logy"]))
            self.xAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["xauto"]))
            self.yAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["yauto"]))
            self.fitIcon	= qt.QIcon(qt.QPixmap(IconDict["fit"]))
            self.searchIcon	= qt.QIcon(qt.QPixmap(IconDict["peaksearch"]))
            self.printIcon	= qt.QIcon(qt.QPixmap(IconDict["fileprint"]))
            self.saveIcon	= qt.QIcon(qt.QPixmap(IconDict["filesave"]))            

    def _build(self):
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self._buildToolBar()
        self._buildGraph()

    def _buildToolBar(self):
        self.toolBar = qt.QWidget(self)
        self.toolBarLayout = qt.QHBoxLayout(self.toolBar)
        self.mainLayout.addWidget(self.toolBar)
        #Autoscale
        self._addToolButton(self.zoomResetIcon,
                            self._zoomReset,
                            'Auto-Scale the Graph')


        #y Autoscale
        self.yAutoScaleButton = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.yAutoScaleButton.setState(qt.QButton.On)
        else:
            self.yAutoScaleButton.setDown(True)

        #x Autoscale
        self.xAutoScaleButton = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.xAutoScaleButton.setState(qt.QButton.On)
        else:
            self.xAutoScaleButton.setDown(True)

        #y Logarithmic
        self.yLogButton = self._addToolButton(self.logyIcon,
                            self._toggleLogY,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.yLogButton.setState(qt.QButton.Off)
        else:
            self.yLogButton.setDown(False)


        #save
        tb = self._addToolButton(self.saveIcon,
                                 self._saveIconSignal,
                                 'Save Active Curve')

        self.toolBarLayout.addWidget(HorizontalSpacer(self.toolBar))
        label=qt.QLabel(self.toolBar)
        label.setText('<b>X:</b>')
        self.toolBarLayout.addWidget(label)

        self._xPos = qt.QLineEdit(self.toolBar)
        self._xPos.setText('------')
        self._xPos.setReadOnly(1)
        self._xPos.setFixedWidth(self._xPos.fontMetrics().width('############'))
        self.toolBarLayout.addWidget(self._xPos)


        label=qt.QLabel(self.toolBar)
        label.setText('<b>Y:</b>')
        self.toolBarLayout.addWidget(label)

        self._yPos = qt.QLineEdit(self.toolBar)
        self._yPos.setText('------')
        self._yPos.setReadOnly(1)
        self._yPos.setFixedWidth(self._yPos.fontMetrics().width('############'))
        self.toolBarLayout.addWidget(self._yPos)
        self.toolBarLayout.addWidget(HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self.printGraph,
                                 'Prints the Graph')

    def _addToolButton(self, icon, action, tip, toggle=None):
        tb      = qt.QToolButton(self.toolBar)            
        if QTVERSION < '4.0.0':
            tb.setIconSet(icon)
            qt.QToolTip.add(tb,tip) 
            if toggle is not None:
                if toggle:
                    tb.setToggleButton(1)
        else:
            tb.setIcon(icon)
            tb.setToolTip(tip)
            if toggle is not None:
                if toggle:
                    tb.setCheckable(1)
        self.toolBarLayout.addWidget(tb)
        self.connect(tb,qt.SIGNAL('clicked()'), action)
        return tb
        

    def _buildGraph(self):
        self.graph = QtBlissGraph.QtBlissGraph(self, uselegendmenu=True)
        self.mainLayout.addWidget(self.graph)


    def setDispatcher(self, w):
        if QTVERSION < '4.0.0':
            self.connect(w, qt.PYSIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.PYSIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.PYSIGNAL("replaceSelection"),
                             self._replaceSelection)
        else:
            self.connect(w, qt.SIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.SIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.SIGNAL("replaceSelection"),
                             self._replaceSelection)
            
    def _addSelection(self, selectionlist):
        if DEBUG:print "_addSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            if not sel.has_key("scanselection"): continue
            if not sel["scanselection"]:continue
            if len(key.split(".")) > 2: continue
            dataObject = sel['dataobject']

            #only one-dimensional selections considered
            if dataObject.info["selectiontype"] != "1D": continue
            
            #there must be something to plot
            if not hasattr(dataObject, 'y'): continue                
            if not hasattr(dataObject, 'x'):
                ylen = len(dataObject.y[0]) 
                if ylen:
                    xdata = Numeric.arange(ylen).astype(Numeric.Float)
                else:
                    #nothing to be plot
                    continue
            if dataObject.x is None:
                ylen = len(dataObject.y[0]) 
                if ylen:
                    xdata = Numeric.arange(ylen).astype(Numeric.Float)
                else:
                    #nothing to be plot
                    continue                    
            elif len(dataObject.x) > 1:
                if DEBUG:print "Mesh plots"
                if not MATPLOTLIB:continue
                else:
                    dataObject.info['legend'] = legend
                    FigureCanvas(dataobject=dataObject,toolbar=True)
                    continue
            else:
                xdata = dataObject.x[0]
            #we have to loop for all y values
            ycounter = -1
            for ydata in dataObject.y:
                ycounter += 1
                newDataObject   = DataObject.DataObject()
                newDataObject.info = copy.deepcopy(dataObject.info)
                #newDataObject.x = [xdata]
                #newDataObject.y = [ydata]
                if dataObject.m is None:
                    mdata = [Numeric.ones(len(ydata)).astype(Numeric.Float)]
                elif len(dataObject.m[0]) > 0:
                    if len(dataObject.m[0]) == len(ydata):
                        index = Numeric.nonzero(dataObject.m[0])
                        if not len(index): continue
                        xdata = Numeric.take(xdata, index)
                        ydata = Numeric.take(ydata, index)
                        mdata = Numeric.take(dataObject.m[0], index)
                    else:
                        raise "ValueError", "Monitor data length different than counter data"
                else:
                    mdata = [Numeric.ones(len(ydata)).astype(Numeric.Float)]
                newDataObject.x = [xdata]
                newDataObject.y = [ydata]
                newDataObject.m = [mdata]
                newDataObject.info['selection'] = copy.deepcopy(sel['selection'])
                ylegend = 'y%d' % ycounter
                if sel['selection'] is not None:
                    if type(sel['selection']) == type({}):
                        if sel['selection'].has_key('x'):
                            #proper scan selection
                            newDataObject.info['selection']['x'] = sel['selection']['x'] 
                            newDataObject.info['selection']['y'] = [sel['selection']['y'][ycounter]]
                            newDataObject.info['selection']['m'] = sel['selection']['m']
                            ilabel = newDataObject.info['selection']['y'][0]
                            ylegend = newDataObject.info['LabelNames'][ilabel]
                newDataObject.info['legend'] = legend + " " + ylegend
                #here I should check the log or linear status
                self.graph.newcurve(newDataObject.info['legend'],
                                    x=xdata,
                                    y=ydata)
                if newDataObject.info['legend'] not in self.dataObjectsList:
                    self.dataObjectsList.append(newDataObject.info['legend'])
                self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
        self.graph.replot()

            
    def _removeSelection(self, selectionlist):
        if DEBUG:print "_removeSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        removelist = []
        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            if not sel.has_key("scanselection"): continue
            if not sel["scanselection"]:continue
            if len(key.split(".")) > 2: continue

            legend = sel['legend'] #expected form sourcename + scan key
            if type(sel['selection']) == type({}):
                if sel['selection'].has_key('y'):
                    if sel['selection'].has_key('cntlist'):
                        for index in sel['selection']['y']:
                            removelist.append(legend +" "+sel['selection']['cntlist'][index])

        if not len(removelist):return
        self.removeCurves(removelist)

    def removeCurves(self, removelist):
        for legend in removelist:
            if legend in self.dataObjectsList:
                del self.dataObjectsList[self.dataObjectsList.index(legend)]
            if legend in self.dataObjectsDict.keys():
                del self.dataObjectsDict[legend]
            self.graph.delcurve(legend)
        self.graph.replot()

    def _replaceSelection(self, selectionlist):
        if DEBUG:print "_replaceSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        doit = 0
        for sel in sellist:
            if not sel.has_key("scanselection"): continue
            if not sel["scanselection"]:continue
            if len(sel["Key"].split(".")) > 2: continue
            dataObject = sel['dataobject']
            if dataObject.info["selectiontype"] == "1D":
                if hasattr(dataObject, 'y'):
                    doit = 1
                    break
        if not doit:return
        self.graph.clearcurves()
        self.graph.replot()
        self.dataObjectsDict={}
        self.dataObjectsList=[]
        self._addSelection(selectionlist)

    def _graphSignalReceived(self, ddict):
        if DEBUG:print "_graphSignalReceived", ddict            
        if ddict['event'] == "MouseAt":
            self._xPos.setText('%.4e' % ddict['x'])
            self._yPos.setText('%.4e' % ddict['y'])
            return
        if ddict['event'] == "SetActiveCurveEvent":
            legend = ddict["legend"]
            if legend is None:return
            splitlegend = legend.split()
            sourcename  = splitlegend[0]
            key         = splitlegend[1]
            ylabel      = splitlegend[2]
            if legend not in self.dataObjectsList:
                if DEBUG:print "unknown legend %s" % legend
                return
            #force the current x label to the appropriate value
            dataObject = self.dataObjectsDict[legend]
            ilabel = dataObject.info['selection']['y'][0]
            ylabel = dataObject.info['LabelNames'][ilabel]
            if len(dataObject.info['selection']['x']):
                ilabel = dataObject.info['selection']['x'][0]
                xlabel = dataObject.info['LabelNames'][ilabel]
            else:
                xlabel = "Point Number"
            self.graph.ylabel(ylabel)
            self.graph.xlabel(xlabel)
            return
            
    def _zoomReset(self):
        if DEBUG:print "_zoomReset"
        self.graph.zoomReset()
        #self.graph.replot()

    def _yAutoScaleToggle(self):
        if DEBUG:print "_yAutoScaleToggle"
        if self.graph.yAutoScale:
            self.graph.yAutoScale = False
            if QTVERSION < '4.0.0':
                self.yAutoScaleButton.setState(qt.QButton.Off)
            else:
                self.yAutoScaleButton.setDown(False)
        else:
            self.graph.yAutoScale = True
            if QTVERSION < '4.0.0':
                self.yAutoScaleButton.setState(qt.QButton.On)
            else:
                self.yAutoScaleButton.setDown(True)
                       
    def _xAutoScaleToggle(self):
        if DEBUG:print "_xAutoScaleToggle"
        if self.graph.xAutoScale:
            self.graph.xAutoScale = False
            if QTVERSION < '4.0.0':
                self.xAutoScaleButton.setState(qt.QButton.Off)
            else:
                self.xAutoScaleButton.setDown(False)
        else:
            self.graph.xAutoScale = True
            if QTVERSION < '4.0.0':
                self.xAutoScaleButton.setState(qt.QButton.On)
            else:
                self.xAutoScaleButton.setDown(True)
                       
                       
    def _toggleLogY(self):
        if DEBUG:print "_toggleLogY"
        self.graph.toggleLogY()
        #self.graph.replot()

    def _saveIconSignal(self):
        if DEBUG:print "_saveIconSignal"


    def printGraph(self):
        if DEBUG:print "printGraphSignal"

        #get graph curvelist (not dataObjects list)??
        legendslist = []
        xdata      = []
        ydata      = []
        xlabel = self.graph.xlabel()
        ylabel = self.graph.ylabel()
        xmin, xmax   = self.graph.getx1axislimits()
        ymin, ymax   = self.graph.gety1axislimits()
        for legend in self.dataObjectsList:
            if legend in self.graph.curves.keys():
                legendslist.append(legend)
                xdata.append(self.dataObjectsDict[legend].x[0])
                ydata.append(self.dataObjectsDict[legend].y[0])
        if len(legendslist):
            FigureCanvas(xlist = xdata, ylist = ydata,
                         xmin = xmin, xmax = xmax,
                         ymin = ymin, ymax = ymax,
                         xlabel = xlabel, ylabel = ylabel,
                         legendslist = legendslist)


if MATPLOTLIB:
    class QMatplotlibCanvas(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
        
    

    class MatplotlibToolbar:
        def __init__(self):
            self._active = None
        
        def set_message(self, msg):
            if DEBUG:print "msg = ",msg

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))

def test():
    w = ScanWindow()
    qt.QObject.connect(app, qt.SIGNAL("lasWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    if QTVERSION < '4.0.0':
        app.setMainWidget(w)
        app.exec_loop()
    else:
        app.exec_()


if __name__ == "__main__":
    test()
