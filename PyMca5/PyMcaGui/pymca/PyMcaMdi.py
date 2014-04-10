#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import sys, getopt, string
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
    
QTVERSION = qt.qVersion()

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from .PyMca_help  import HelpDict
DEBUG = 0

__version__ = "1.5"

class PyMcaMdi(qt.QMainWindow):
    def __init__(self, parent=None, name="PyMca Mdi", fl=None, options={}):
        qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle(name)
        if fl is None:
            fl = qt.Qt.WA_DeleteOnClose

        self.options= {}
        self.options["FileToolBar"]= options.get("FileToolBar", 1)
        self.options["WinToolBar"] = options.get("WinToolBar", 1)
        self.options["MenuFile"]   = options.get("MenuFile", 1)
        self.options["MenuTools"]  = options.get("MenuTools", 1)
        self.options["MenuWindow"] = options.get("MenuWindow", 1)
        self.options["MenuHelp"]   = options.get("MenuHelp", 1)

        self.splitter = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Horizontal)
        #self.splitterLayout = qt.QVBoxLayout(self.splitter)
        #self.splitterLayout.setContentsMargins(0, 0, 0, 0)
        #self.splitterLayout.setSpacing(0)
        
        self.printer= qt.QPrinter()
        self.mdi = qt.QWorkspace(self.splitter)
        self.mdi.setScrollBarsEnabled(1)
        #if QTVERSION > '4.0.0':self.mdi.setBackground(qt.QBrush(qt.QColor(238,234,238)))
        #self.setCentralWidget(self.mdi)
        #self.splitterLayout.addWidget(self.mdi)
            
        self.setCentralWidget(self.splitter)

        self.splitter.insertWidget(1, self.mdi)
        self.windowMapper = qt.QSignalMapper(self)
        self.connect(self.windowMapper, qt.SIGNAL("mapped(QWidget *)"),
                     self.mdi, qt.SLOT("setActiveWindow(QWidget *)"))


        #self.setDockEnabled(qt.Qt.DockTop, 0)

        

        self.initIcons()
        if QTVERSION > '4.0.0':
            self.createActions()

        self.initMenuBar()
        self.initToolBar()

        self.followActiveWindow= 0

        self.mdi.show()
        #self.resize(600,400)
        
    def createActions(self):
            #fileopen
            self.actionOpen = qt.QAction(self)
            self.actionOpen.setText(QString("&Open"))
            self.actionOpen.setIcon(self.Icons["fileopen"])
            self.actionOpen.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
            self.connect(self.actionOpen, qt.SIGNAL("triggered(bool)"),
                         self.onOpen)
            #filesaveas
            self.actionSaveAs = qt.QAction(self)
            self.actionSaveAs.setText(QString("&Save"))
            self.actionSaveAs.setIcon(self.Icons["filesave"])
            self.actionSaveAs.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
            self.connect(self.actionSaveAs, qt.SIGNAL("triggered(bool)"),
                         self.onSaveAs)

            #filesave
            self.actionSave = qt.QAction(self)
            self.actionSave.setText(QString("Save &Defaults"))
            #self.actionSave.setIcon(self.Icons["filesave"])
            #self.actionSave.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
            self.connect(self.actionSave, qt.SIGNAL("triggered(bool)"),
                         self.onSave)

            #fileprint
            self.actionPrint = qt.QAction(self)
            self.actionPrint.setText(QString("&Print"))
            self.actionPrint.setIcon(self.Icons["fileprint"])
            self.actionPrint.setShortcut(qt.Qt.CTRL+qt.Qt.Key_P)
            self.connect(self.actionPrint, qt.SIGNAL("triggered(bool)"),
                         self.onPrint)

            #filequit
            self.actionQuit = qt.QAction(self)
            self.actionQuit.setText(QString("&Quit"))
            #self.actionQuit.setIcon(self.Icons["fileprint"])
            self.actionQuit.setShortcut(qt.Qt.CTRL+qt.Qt.Key_Q)
            qApp = qt.QApplication.instance() 
            self.actionQuit.triggered.connect(qApp.closeAllWindows)
                               #qt.SIGNAL("triggered(bool)"),
                               #qt.qApp,
                               #qt.SLOT("closeAllWindows()"))

    def initIcons(self):
        self.Icons= {}
        for (name, icon) in IconDict.items():
            pixmap= qt.QPixmap(icon)
            self.Icons[name]= qt.QIcon(pixmap)

    def initToolBar(self):
        if self.options["FileToolBar"]:
            self.fileToolBar= self.addToolBar("filetoolbar")
            self.fileToolBar.addAction(self.actionOpen)
            self.fileToolBar.addAction(self.actionSaveAs)
            self.fileToolBar.addAction(self.actionPrint)
            self.fileToolBar.addSeparator()

        self.onInitToolBar()

        if self.options["WinToolBar"]:
            self.winToolBar = self.addToolBar("wintoolbar")

    def onWinToolMenu(self, idx):
        if DEBUG:
            print("onWinToolMenu %d " % idx)
        for midx in self.winToolMenuIndex:
                self.winToolMenu.setItemChecked(midx, midx==idx)
        act= self.winToolMenuIndex.index(idx)
        self.winToolButton.setTextLabel(self.winToolMenuText[act])
        if act==0:
            self.winToolMenuAction= self.windowCascade
        elif act==1:
            self.winToolMenuAction= self.windowTile
        elif act==2:
            self.winToolMenuAction= self.windowHorizontal
        elif act==3:
            self.winToolMenuAction= self.windowVertical
        self.onWinToolAction()

    def onWinToolAction(self):
            apply(self.winToolMenuAction, ())

    #
    # Mdi windows geometry
    #
    def windowCascade(self):
        if self.followActiveWindow: self.__disconnectFollow()
        self.mdi.cascade()
        for window in self.mdi.windowList():
                window.resize(0.7*self.mdi.width(),0.7*self.mdi.height())
        if self.followActiveWindow: self.__connectFollow()

    def windowTile(self):
        if self.followActiveWindow: self.__disconnectFollow()
        self.mdi.tile()
        if self.followActiveWindow: self.__connectFollow()

    def windowHorizontal(self):
        #if self.followActiveWindow: self.__disconnectFollow()
        if not len(self.mdi.windowList()): return
        windowheight=float(self.mdi.height())/len(self.mdi.windowList())
        i=0
        for window in self.mdi.windowList():
                window.parentWidget().showNormal()
                window.parentWidget().setGeometry(0, int(windowheight*i),
                                        self.mdi.width(),int(windowheight))
                if QTVERSION < '4.0.0':
                    window.parentWidget().raiseW()
                else:
                    window.parentWidget().raise_()
                i+=1
        self.mdi.update()
        self.update()
        #if self.followActiveWindow: self.__connectFollow()

    def windowVertical(self):
        #if self.followActiveWindow: self.__disconnectFollow()
        if not len(self.mdi.windowList()): return
        windowwidth=float(self.mdi.width())/len(self.mdi.windowList())
        i=0
        for window in self.mdi.windowList():
                window.parentWidget().showNormal()
                window.parentWidget().setGeometry(int(windowwidth*i),0,
                                        int(windowwidth),self.mdi.height())
                
                if QTVERSION < '4.0.0':
                    window.parentWidget().raiseW()
                else:
                    window.parentWidget().raise_()
                i+=1
        self.mdi.update()
        self.update()
        #if self.followActiveWindow: self.__connectFollow()

    def windowFullScreen(self):
        if len(self.mdi.windowList()):
                self.mdi.activeWindow().showMaximized()


    def initMenuBar(self):
        if self.options["MenuFile"]:
            #self.menubar = qt.QMenuBar(self)
            self.menuFile= qt.QMenu()
            self.menuFile.addAction(self.actionOpen)
            self.menuFile.addAction(self.actionSaveAs)
            self.menuFile.addAction(self.actionSave)
            self.menuFile.addSeparator()
            self.menuFile.addAction(self.actionPrint)
            self.menuFile.addSeparator()
            self.menuFile.addAction(self.actionQuit)
            self.menuFile.setTitle("&File")
            self.menuBar().addMenu(self.menuFile)
            self.onInitMenuBar(self.menuBar())

        if self.options["MenuTools"]:
            self.menuTools= qt.QMenu()
            #self.menuTools.setCheckable(1)
            self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"),
                         self.menuToolsAboutToShow)
            self.menuTools.setTitle("&Tools")
            self.menuBar().addMenu(self.menuTools)
                
        if self.options["MenuWindow"]:
            self.menuWindow= qt.QMenu()
            #self.menuWindow.setCheckable(1)
            self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
            self.menuWindow.setTitle("&Window")
            self.menuBar().addMenu(self.menuWindow)

        if self.options["MenuHelp"]:
            self.menuHelp= qt.QMenu()
            self.menuHelp.addAction("&About", self.onAbout)
            self.menuHelp.addAction("About &Qt",self.onAboutQt)
            self.menuBar().addSeparator()
            self.menuHelp.setTitle("&Help")
            self.menuBar().addMenu(self.menuHelp)

    def menuWindowAboutToShow(self):
        if DEBUG:
            print("menuWindowAboutToShow")
        self.menuWindow.clear()
        if len(self.mdi.windowList())==0:
            return
        self.menuWindow.addAction("&Cascade", self.windowCascade)
        self.menuWindow.addAction("&Tile", self.windowTile)
        self.menuWindow.addAction("&Tile Horizontally", self.windowHorizontal)
        self.menuWindow.addAction("&Tile Vertically", self.windowVertical)
        windows=self.mdi.windowList()
        if len(windows) > 0:
            self.menuWindow.addSeparator()
        num = 0
        for window in windows:
            text = "&%d %s"%(num, str(window.windowTitle()))
            num += 1                
            action = self.menuWindow.addAction(text)
            action.setCheckable(True)
            action.setChecked(window == self.mdi.activeWindow())
            self.connect(action, qt.SIGNAL("triggered()"),
                         self.windowMapper, qt.SLOT("map()"))
            self.windowMapper.setMapping(action, window)

    def menuWindowActivated(self, idx = None):
        if DEBUG:
            print("menuWindowActivated idx = ",idx)
        if idx is None:return
        if self.menuWindowMap[idx].isHidden():
            self.menuWindowMap[idx].show()
            self.menuWindowMap[idx].raise_()
        self.menuWindowMap[idx].setFocus()

    def __connectFollow(self):
        self.connect(self.mdi, qt.SIGNAL("windowActivated(QWidget*)"), self.onWindowActivated)

    def __disconnectFollow(self):
        self.disconnect(self.mdi, qt.SIGNAL("windowActivated(QWidget*)"), self.onWindowActivated)

    def setFollowActiveWindow(self, follow):
        if follow != self.followActiveWindow:
            if not follow:
                self.__disconnectFollow()
            else:
                self.__connectFollow()
            self.followActiveWindow= follow

    def onWindowActivated(self, win):
        print("Window activated")
        pass

    #
    # Dock windows
    #
    def isCustomizable(self):
        nb= 0
        for win in self.dockWindows():
                nb += isinstance(win, DockWindow)
        return (nb>0)

    def customize(self, *args):
        dg= DockPlaceDialog(self, window=self, title="Tool Places")
        dg.exec_loop()

    #
    # Menus customization
    #
    def onInitMenuBar(self, menubar):
        pass

    def onInitFileToolBar(self, toolbar):
        pass

    def onInitToolBar(self):
        pass

    def onInitWinToolBar(self, toolbar):
        pass

    def menuToolsAboutToShow(self):
        if DEBUG:
            print("menuToolsAboutToShow")
        self.menuTools.clear()
        self.menuToolsMap= {}
        """
        for win in self.dockWindows():
            if isinstance(win, DockWindow):
                    idx= self.menuTools.insertItem("%s"%str(win.caption()), self.menuToolsActivated)
                    self.menuToolsMap[idx]= win
                    self.menuTools.setItemChecked(idx, not win.isHidden())
        """
        if len(self.menuToolsMap.keys()):
            self.menuTools.insertSeparator()
            self.menuTools.insertItem("Customize", self.customize)

    def menuToolsActivated(self, idx):
        if DEBUG:
            print("menuToolsActivated idx = ",idx)
        if self.menuTools.isItemChecked(idx):
            self.menuToolsMap[idx].hide()
        else:
            self.menuToolsMap[idx].show()



    #
    # Menus customization
    #
    def onInitMenuBar(self, menubar):
        pass

    def onInitFileToolBar(self, toolbar):
        pass

    def onInitToolBar(self):
        pass

    def onInitWinToolBar(self, toolbar):
        pass

    #
    # Menus callback
    #
    def onAboutQt(self):
        qt.QMessageBox.aboutQt(self, "About Qt")

    def onAbout(self):
        qt.QMessageBox.about(self, "MDI",
                "MDI Application Framework\nVersion: "+__version__)

    def onOpen(self):
        qt.QMessageBox.about(self, "Open", "Not implemented")

    def onSave(self):
        qt.QMessageBox.about(self, "Save", "Not implemented")

    def onSaveAs(self):
        qt.QMessageBox.about(self, "SaveAs", "Not implemented")

    def onPrint(self):
        qt.QMessageBox.about(self, "Print", "Not implemented")





def main(args):
    app = qt.QApplication(args)
    #if sys.platform == 'win32':
    if 1:
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)

    options     = ''
    longoptions = ['spec=','shm=']
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except getopt.error:
        print(sys.exc_info()[1])
        sys.exit(1)
    # --- waiting widget
    kw={}
    for opt, arg in opts:
        if  opt in ('--spec'):
            kw['spec'] = arg
        elif opt in ('--shm'):
            kw['shm']  = arg
    #demo = McaWindow.McaWidget(**kw)
    demo = PyMca()
    app.lastWindowClosed.connect(app.quit())
    demo.show()
    app.exec_()

if __name__ == '__main__':
    main(sys.argv)

