#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
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
import Elements
import XRayTubeEbel
import numpy.oldnumeric as Numeric
import QtBlissGraph
qt = QtBlissGraph.qt

DEBUG = 0

if qt.qVersion() > '4.0.0':
    class QGridLayout(qt.QGridLayout):
        def addMultiCellWidget(self, w, r0, r1, c0, c1, *var):
            self.addWidget(w, r0, c0, 1 + r1 - r0, 1 + c1 - c0)

class QXTube(qt.QWidget):
    def __init__(self, parent=None, initdict = None):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, "TubeWidget",0)
        else:
            qt.QWidget.__init__(self, parent)

        self.l = qt.QVBoxLayout(self)
        self.l.setMargin(0)
        self.l.setSpacing(0)
        
        self.tubeWidget = TubeWidget(self, initdict = initdict)
        self.setParameters = self.tubeWidget.setParameters
        self.getParameters = self.tubeWidget.getParameters

        label = qt.QLabel(self)

        
        hbox = qt.QWidget(self)
        hboxl = qt.QHBoxLayout(hbox)
        hboxl.setMargin(0)
        hboxl.setSpacing(0)
        self.plotButton = qt.QPushButton(hbox)
        self.plotButton.setText("Plot Continuum")

        self.exportButton = qt.QPushButton(hbox)
        self.exportButton.setText("Export to Fit")

        #grid.addWidget(self.plotButton, 7, 1)
        #grid.addWidget(self.exportButton, 7, 3)
        
        hboxl.addWidget(self.plotButton)
        hboxl.addWidget(self.exportButton)

        self.l.addWidget(self.tubeWidget)

        f = label.font()
        f.setItalic(1)
        label.setFont(f)
        label.setAlignment(qt.Qt.AlignRight)
        label.setText("H. Ebel, X-Ray Spectrometry 28 (1999) 255-266    ")
        self.l.addWidget(label)
        
        self.l.addWidget(hbox)
        self.graph = None
        
        self.connect(self.plotButton,
                     qt.SIGNAL("clicked()"),
                     self.plot)

        self.connect(self.exportButton,
                     qt.SIGNAL("clicked()"),
                     self._export)

    def plot(self):
        d = self.tubeWidget.getParameters()
        transmission    = d["transmission"]
        anode           = d["anode"]
        anodedensity    = d["anodedensity"]
        anodethickness  = d["anodethickness"]
        
        voltage         = d["voltage"]
        wele            = d["window"]
        wdensity        = d["windowdensity"]
        wthickness      = d["windowthickness"]
        alphae          = d["alphae"]
        alphax          = d["alphax"]

        delta           = d["deltaplotting"]
        e = Numeric.arange(1, voltage, delta)

        if self.graph is None:
            self.graph = QtBlissGraph.QtBlissGraph(self)
            self.l.addWidget(self.graph)
            #self.graph.setTitle("Reference: X-Ray Spectrometry 28 (1999) 255-256")
            self.graph.xlabel("Energy (keV)")
            self.graph.ylabel("photons/sr/mA/keV/s")
            self.graph.show()

        if __name__ == "__main__":
            continuumR = XRayTubeEbel.continuumEbel([anode, anodedensity, anodethickness],
                                             voltage, e,
                                             [wele, wdensity, wthickness],
                                             alphae=alphae, alphax=alphax,
                                             transmission=0,
                                             targetthickness=anodethickness)

            continuumT = XRayTubeEbel.continuumEbel([anode, anodedensity, anodethickness],
                                             voltage, e,
                                             [wele, wdensity, wthickness],
                                             alphae=alphae, alphax=alphax,
                                             transmission=1,
                                             targetthickness=anodethickness)




            self.graph.newcurve("continuumR", e, continuumR)
            self.graph.newcurve("continuumT", e, continuumT)
        else:
            continuum = XRayTubeEbel.continuumEbel([anode, anodedensity, anodethickness],
                                             voltage, e,
                                             [wele, wdensity, wthickness],
                                             alphae=alphae, alphax=alphax,
                                             transmission=transmission,
                                             targetthickness=anodethickness)
            self.graph.newcurve("continuum", e, continuum)

        self.graph.zoomReset()
        self.graph.replot()

    def _export(self):
        d = self.tubeWidget.getParameters()
        transmission    = d["transmission"]
        anode           = d["anode"]
        anodedensity    = d["anodedensity"]
        anodethickness  = d["anodethickness"]
        
        voltage         = d["voltage"]
        wele            = d["window"]
        wdensity        = d["windowdensity"]
        wthickness      = d["windowthickness"]
        alphae          = d["alphae"]
        alphax          = d["alphax"]
        delta           = d["deltaplotting"]

        e = Numeric.arange(1, voltage, delta)

        d["event"]      = "TubeUpdated"
        d["energyplot"] = e
        d["continuum"]  = XRayTubeEbel.continuumEbel([anode, anodedensity, anodethickness],
                                             voltage, e,
                                             [wele, wdensity, wthickness],
                                             alphae=alphae, alphax=alphax,
                                             transmission=transmission,
                                             targetthickness=anodethickness)
        

        fllines = XRayTubeEbel.characteristicEbel([anode, anodedensity, anodethickness],
                     voltage,
                     [wele, wdensity, wthickness],
                     alphae=alphae, alphax=alphax,
                     transmission=transmission,
                     targetthickness=anodethickness)

        d["characteristic"] = fllines
        if DEBUG:
            fsum = 0.0
            for l in fllines:
                print "%s %.4f %.3e" % (l[2],l[0],l[1])
                fsum += l[1]
            print fsum

        energy, energyweight, energyscatter = XRayTubeEbel.generateLists([anode, anodedensity,
                                                                          anodethickness],
                                                        voltage,
                                                        window = [wele, wdensity, wthickness],
                                                        alphae = alphae, alphax = alphax,
                                                        transmission = transmission,
                                                        targetthickness=anodethickness)

        d["energylist"]        = energy
        d["weightlist"]  = energyweight
        d["scatterlist"] = energyscatter
        d["flaglist"]    = Numeric.ones(len(energy))

        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("QXTubeSignal"), (d,))
        else:
            self.emit(qt.SIGNAL("QXTubeSignal"), d)
            

class TubeWidget(qt.QWidget):
    def __init__(self, parent=None, initdict = None):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, "TubeWidget",0)
        else:
            qt.QWidget.__init__(self, parent)
        self._build()
        if qt.qVersion() < '4.0.0':
            self.connect(self.anodeCombo, qt.PYSIGNAL("MyQComboBoxSignal"),
                         self._anodeSlot)
            self.connect(self.windowCombo, qt.PYSIGNAL("MyQComboBoxSignal"),
                         self._windowSlot)
        else:            
            self.connect(self.anodeCombo, qt.SIGNAL("MyQComboBoxSignal"),
                         self._anodeSlot)
            self.connect(self.windowCombo, qt.SIGNAL("MyQComboBoxSignal"),
                         self._windowSlot)
        self.connect(self.transmissionCheckBox,
                     qt.SIGNAL("clicked()"),
                     self._transmissionSlot)
            
        if initdict is not None:
            self.setParameters(initdict)
        else:
            d = {}
            d["transmission"]    = 0
            d["voltage"]         = 30.0
            d["anode"]           = "Ag"
            d["anodethickness"]  = 0.0002
            d["anodedensity"]    = Elements.Element["Ag"]["density"]
            d["window"]          = "Be"
            d["windowthickness"] = 0.0125
            d["windowdensity"]   = Elements.Element["Be"]["density"]
            d["alphax"]          = 90.0
            d["alphae"]          = 90.0
            d["deltaplotting"]   = 0.10
            self.setParameters(d)

    def _build(self):
        layout = qt.QVBoxLayout(self)
        layout.setMargin(11)

        gridwidget   = qt.QWidget(self)
        if qt.qVersion() < '4.0.0':
            grid = qt.QGridLayout(gridwidget, 8, 4, 0, 6)
        else:
            grid = QGridLayout(gridwidget)
            grid.setMargin(0)
            grid.setSpacing(6)


        self.transmissionCheckBox = qt.QCheckBox(gridwidget)
        self.transmissionCheckBox.setText("Transmission Tube")

        voltage = qt.QLabel(gridwidget)
        voltage.setText("Voltage")

        self.voltage = qt.QLineEdit(gridwidget)
        
        grid.addMultiCellWidget(self.transmissionCheckBox, 0 ,0, 0, 1)
        grid.addWidget(voltage, 0 ,2)
        grid.addWidget(self.voltage, 0 ,3)


        #materials            
        mlabel               = qt.QLabel(gridwidget)
        mlabel.setText("Material")
        dlabel               = qt.QLabel(gridwidget)
        dlabel.setText("Density")
        tlabel               = qt.QLabel(gridwidget)
        tlabel.setText("Thickness")



        #anode            
        anodelabel = qt.QLabel(gridwidget)
        anodelabel.setText("Anode")
        self.anodeCombo     = MyQComboBox(gridwidget, options = Elements.ElementList)
        self.anodeDensity   = qt.QLineEdit(gridwidget)
        self.anodeThickness = qt.QLineEdit(gridwidget)
        
        #window
        windowlabel = qt.QLabel(gridwidget)
        windowlabel.setText("Window")
        self.windowCombo     = MyQComboBox(gridwidget,  options = Elements.ElementList)
        self.windowDensity   = qt.QLineEdit(gridwidget)
        self.windowThickness = qt.QLineEdit(gridwidget)

        grid.addWidget(mlabel, 1 ,1)
        grid.addWidget(dlabel, 1 ,2)
        grid.addWidget(tlabel, 1 ,3)
    
        grid.addWidget(anodelabel,          2, 0)
        grid.addWidget(self.anodeCombo,     2, 1)
        grid.addWidget(self.anodeDensity,   2, 2)
        grid.addWidget(self.anodeThickness, 2, 3)
        
        grid.addWidget(windowlabel,          3, 0)
        grid.addWidget(self.windowCombo,     3, 1)
        grid.addWidget(self.windowDensity,   3, 2)
        grid.addWidget(self.windowThickness, 3, 3)

        #angles
        alphaelabel = qt.QLabel(gridwidget)
        alphaelabel.setText("Alpha electron")
        self.alphaE = qt.QLineEdit(gridwidget)
        alphaxlabel = qt.QLabel(gridwidget)
        alphaxlabel.setText("Alpha x-ray")
        self.alphaX = qt.QLineEdit(gridwidget)

        grid.addWidget(alphaelabel, 4, 2)
        grid.addWidget(self.alphaE, 4, 3)
        grid.addWidget(alphaxlabel, 5, 2)
        grid.addWidget(self.alphaX, 5, 3)

        #delta energy
        deltalabel = qt.QLabel(gridwidget)
        deltalabel.setText("Delta energy (keV) just for plotting")
        self.delta = qt.QLineEdit(gridwidget)

        grid.addMultiCellWidget(deltalabel, 6, 6, 0, 3)
        grid.addWidget(self.delta, 6, 3)

        layout.addWidget(gridwidget)
        
    def setParameters(self, d):
        """
            d["transmission"]    = 1
            d["anode"]           = "Ag"
            d["anodethickness"]  = 0.0002
            d["anodedensity"]    = None
            d["window"]          = "Be"
            d["windowthickness"] = 0.0125
            d["windowdensity"]   = None
            d["anglex"]          = 90.0
            d["anglee"]          = 90.0
            d["deltaplotting"]   = 0.2
        """
        if d.has_key("transmission"):
            if d["transmission"]:
                self.transmissionCheckBox.setChecked(1)
            else:
                self.transmissionCheckBox.setChecked(0)
            self._transmissionSlot()                
        if d.has_key("voltage"): self.voltage.setText("%.1f" % d["voltage"])
        if d.has_key("anode"):
            self.anodeCombo.setCurrentIndex(Elements.ElementList.index(d["anode"]))
            self.anodeDensity.setText("%f" % Elements.Element[d["anode"]]["density"])
        if d.has_key("window"):
            self.windowCombo.setCurrentIndex(Elements.ElementList.index(d["window"]))
            self.windowDensity.setText("%f" % Elements.Element[d["window"]]["density"])
        if d.has_key("anodethickness"): self.anodeThickness.setText("%f" % d["anodethickness"])
        if d.has_key("anodedensity"): self.anodeDensity.setText("%f" % d["anodedensity"])
        if d.has_key("windowthickness"): self.windowThickness.setText("%f" % d["windowthickness"])
        if d.has_key("windowdensity"): self.windowDensity.setText("%f" % d["windowdensity"])
        if d.has_key("alphax"):  self.alphaX.setText("%.1f" % d["alphax"])
        if d.has_key("alphae"):  self.alphaE.setText("%.1f" % d["alphae"])
        if d.has_key("deltaplotting"):   self.delta.setText("%.3f" % d["deltaplotting"])

    def getParameters(self):
        d = {}
        if self.transmissionCheckBox.isChecked():
            d["transmission"]    = 1
        else:
            d["transmission"]    = 0
        d["voltage"] = float(str(self.voltage.text()))
        d["anode"] = self.anodeCombo.getCurrent()[1]
        d["anodethickness"]  = float(str(self.anodeThickness.text()))
        d["anodedensity"]    = float(str(self.anodeDensity.text()))
        d["window"] = self.windowCombo.getCurrent()[1]
        d["windowthickness"]  = float(str(self.windowThickness.text()))
        d["windowdensity"]    = float(str(self.windowDensity.text()))
        d["alphax"]          = float(str(self.alphaX.text()))
        d["alphae"]          = float(str(self.alphaE.text()))
        d["deltaplotting"]   = float(str(self.delta.text()))
        return d

    def _anodeSlot(self, ddict):
        if DEBUG:print "_anodeSlot", ddict
        self.anodeDensity.setText("%f" % Elements.Element[ddict["element"]]["density"])
        
    def _windowSlot(self, ddict):
        if DEBUG:print "_windowSlot", ddict
        self.windowDensity.setText("%f" % Elements.Element[ddict["element"]]["density"])

    def _transmissionSlot(self):
        if DEBUG:print "_transmissionSlot"
        if self.transmissionCheckBox.isChecked():
            self.anodeThickness.setEnabled(1)
        else:
            self.anodeThickness.setEnabled(0)

class MyQComboBox(qt.QComboBox):
    def __init__(self,parent = None,name = None,fl = 0,
                 options=['1','2','3'],row=None,col=None):
        if row is None: row = 0
        if col is None: col = 0
        self.row = row
        self.col = col
        qt.QComboBox.__init__(self,parent)
        self.setOptions(options)
        self.setDuplicatesEnabled(False)
        self.setEditable(False)
        self.connect(self, qt.SIGNAL("activated(const QString &)"),self._mySignal)

    if qt.qVersion() < '4.0.0':
        def setCurrentIndex(self, index):
            return self.setCurrentItem(index)

        def currentIndex(self):
            return self.currentItem()

    def setOptions(self,options=['1','2','3']):
        self.clear()
        if qt.qVersion() < '4.0.0':
            self.insertStrList(options)
        else:
            for item in options:
                self.addItem(item)
                
    def getCurrent(self):
        return   self.currentIndex(),str(self.currentText())

    def _mySignal(self, qstring0):
        if DEBUG:print "_mySignal ", qstring0
        text = str(qstring0)
        d = {}
        d['event']   = 'activated'
        d['element'] = text
        #d['z'] = Elemens.ElementList.index(d) + 1
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('MyQComboBoxSignal'),(d,))
        else:
            self.emit(qt.SIGNAL('MyQComboBoxSignal'), (d))
        
if __name__ == "__main__":
    app = qt.QApplication([])

    w = QXTube()
    w.show()

    if qt.qVersion() < '4.0.0':
        qt.QObject.connect(app,
                           qt.SIGNAL("lastWindowClosed()"),
                           app,
                           qt.SLOT("quit()"))
        app.exec_loop()
    else:
        app.exec_()
        
