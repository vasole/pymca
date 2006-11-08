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
#!/usr/bin/env python
import sys, getopt, string
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
    except:
        import qt
else:
    import qt
    
QTVERSION = qt.qVersion()

from PyMca_Icons import IconDict
from PyMca_help  import HelpDict
DEBUG = 0

__version__ = "1.5"

class PyMca(qt.QMainWindow):
    def __init__(self, parent=None, name="PyMca Mdi", fl=None, options={}):
        if QTVERSION < '4.0.0':
            if fl is None: fl=qt.Qt.WDestructiveClose
            qt.QMainWindow.__init__(self, parent, name, fl)
            self.setCaption(name)
        else:
            qt.QMainWindow.__init__(self, parent)
            self.setWindowTitle(name)
            if fl is None: fl = qt.Qt.WA_DeleteOnClose

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
        #self.splitterLayout.setMargin(0)
        #self.splitterLayout.setSpacing(0)
        
        self.printer= qt.QPrinter()
        if qt.qVersion()>="3.0.0":
                self.mdi= qt.QWorkspace(self.splitter)
                self.mdi.setScrollBarsEnabled(1)
                #if QTVERSION > '4.0.0':self.mdi.setBackground(qt.QBrush(qt.QColor(238,234,238)))
                #self.setCentralWidget(self.mdi)
        else:
                self.mdi= qt.QWorkspace(self.splitter)
                self.mdi.setBackgroundColor(qt.QColor("gainsboro"))
                #self.setCentralWidget(self.mdi)
        #self.splitterLayout.addWidget(self.mdi)
            
        self.setCentralWidget(self.splitter)
        if QTVERSION < '4.0.0':
            self.splitter.moveToLast(self.mdi)
        else:
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
        
    if QTVERSION > '4.0.0':
        def createActions(self):
                #fileopen
                self.actionOpen = qt.QAction(self)
                self.actionOpen.setText(qt.QString("&Open"))
                self.actionOpen.setIcon(self.Icons["fileopen"])
                self.actionOpen.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
                self.connect(self.actionOpen, qt.SIGNAL("triggered(bool)"),
                             self.onOpen)
                #filesaveas
                self.actionSaveAs = qt.QAction(self)
                self.actionSaveAs.setText(qt.QString("&Save"))
                self.actionSaveAs.setIcon(self.Icons["filesave"])
                self.actionSaveAs.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSaveAs, qt.SIGNAL("triggered(bool)"),
                             self.onSaveAs)

                #filesave
                self.actionSave = qt.QAction(self)
                self.actionSave.setText(qt.QString("Save &Defaults"))
                #self.actionSave.setIcon(self.Icons["filesave"])
                #self.actionSave.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSave, qt.SIGNAL("triggered(bool)"),
                             self.onSave)

                #fileprint
                self.actionPrint = qt.QAction(self)
                self.actionPrint.setText(qt.QString("&Print"))
                self.actionPrint.setIcon(self.Icons["fileprint"])
                self.actionPrint.setShortcut(qt.Qt.CTRL+qt.Qt.Key_P)
                self.connect(self.actionPrint, qt.SIGNAL("triggered(bool)"),
                             self.onPrint)

                #filequit
                self.actionQuit = qt.QAction(self)
                self.actionQuit.setText(qt.QString("&Quit"))
                #self.actionQuit.setIcon(self.Icons["fileprint"])
                self.actionQuit.setShortcut(qt.Qt.CTRL+qt.Qt.Key_Q)
                qt.QObject.connect(self.actionQuit,
                                   qt.SIGNAL("triggered(bool)"),
                                   qt.qApp,
                                   qt.SLOT("closeAllWindows()"))

    def initIcons(self):
        self.Icons= {}
        for (name, icon) in IconDict.items():
            pixmap= qt.QPixmap(icon)
            if QTVERSION < '4.0.0':
                self.Icons[name]= qt.QIconSet(pixmap)
            else:
                self.Icons[name]= qt.QIcon(pixmap)

    def initToolBar(self):
        if self.options["FileToolBar"]:
            if QTVERSION < '4.0.0':
                self.fileToolBar= qt.QToolBar(self, "filetoolbar")
                self.fileToolBar.setLabel("File Operations")
                self.onInitFileToolBar(self.fileToolBar)
                fileOpen= qt.QToolButton(self.Icons["fileopen"], "Open", qt.QString.null, self.onOpen, self.fileToolBar, "open")
                qt.QWhatsThis.add(fileOpen, HelpDict["fileopen"])
                fileSave= qt.QToolButton(self.Icons["filesave"], "Save As", qt.QString.null, self.onSaveAs, self.fileToolBar, "save")
                qt.QWhatsThis.add(fileSave, HelpDict["filesave"])
                filePrint= qt.QToolButton(self.Icons["fileprint"], "Print", qt.QString.null, self.onPrint, self.fileToolBar, "print")
                qt.QWhatsThis.add(filePrint, HelpDict["fileprint"])
                self.fileToolBar.addSeparator()
                qt.QWhatsThis.whatsThisButton(self.fileToolBar)
            else:
                self.fileToolBar= self.addToolBar("filetoolbar")
                self.fileToolBar.addAction(self.actionOpen)
                self.fileToolBar.addAction(self.actionSaveAs)
                self.fileToolBar.addAction(self.actionPrint)
                self.fileToolBar.addSeparator()

        self.onInitToolBar()

        if self.options["WinToolBar"]:
                if QTVERSION < '4.0.0':
                    self.winToolBar= qt.QToolBar(self, "wintoolbar")
                    self.winToolBar.setLabel("Window resize")
                    self.onInitWinToolBar(self.winToolBar)
                    FullScreen= qt.QToolButton(self.Icons["window_fullscreen"], "Full Screen", qt.QString.null,
                                    self.windowFullScreen, self.winToolBar, "fullscreen")
                    qt.QWhatsThis.add(FullScreen, HelpDict["fullscreen"])
                    self.winToolButton= qt.QToolButton(self.Icons["window_nofullscreen"], "Tile",
                                                    qt.QString.null, self.onWinToolAction, self.winToolBar, "wintile")
                    qt.QWhatsThis.add(self.winToolButton, HelpDict["nofullscreen"])
                    self.winToolMenu= qt.QPopupMenu(self.winToolButton)
                    self.winToolMenu.setCheckable(1)
                    self.winToolMenuText= ["Cascade", "Tile", "Tile Horizontally", "Tile Vertically"]
                    self.winToolMenuIndex= []
                    for text in self.winToolMenuText:
                            self.winToolMenuIndex.append(self.winToolMenu.insertItem(text, self.onWinToolMenu))

                    self.winToolMenu.setItemChecked(self.winToolMenuIndex[1], 1)
                    self.winToolMenuAction= self.windowTile

                    self.winToolButton.setPopup(self.winToolMenu)
                    self.winToolButton.setPopupDelay(0)
                else:
                    self.winToolBar = self.addToolBar("wintoolbar")

    def onWinToolMenu(self, idx):
        if DEBUG:
            print "onWinToolMenu ",idx
        for midx in self.winToolMenuIndex:
                self.winToolMenu.setItemChecked(midx, midx==idx)
        act= self.winToolMenuIndex.index(idx)
        self.winToolButton.setTextLabel(self.winToolMenuText[act])
        if act==0:      self.winToolMenuAction= self.windowCascade
        elif act==1:    self.winToolMenuAction= self.windowTile
        elif act==2:    self.winToolMenuAction= self.windowHorizontal
        elif act==3:    self.winToolMenuAction= self.windowVertical
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
            if QTVERSION < '4.0.0':
                self.menuFile= qt.QPopupMenu(self.menuBar())
                idx= self.menuFile.insertItem(self.Icons["fileopen"],
                                              qt.QString("&Open"),
                                              self.onOpen,
                                              qt.Qt.CTRL+qt.Qt.Key_O)
                self.menuFile.setWhatsThis(idx, HelpDict["fileopen"])
                idx= self.menuFile.insertItem(self.Icons["filesave"], "&Save",
                                              self.onSave,
                                              qt.Qt.CTRL+qt.Qt.Key_S)
                self.menuFile.setWhatsThis(idx, HelpDict["filesave"])
                self.menuFile.insertItem("Save &as", self.onSaveAs)
                self.menuFile.insertSeparator()
                idx= self.menuFile.insertItem(self.Icons["fileprint"],
                                              "&Print", self.onPrint,
                                              qt.Qt.CTRL+qt.Qt.Key_P)
                self.menuFile.setWhatsThis(idx, HelpDict["fileprint"])
                self.menuFile.insertSeparator()
                self.menuFile.insertItem("&Quit", qt.qApp,
                                         qt.SLOT("closeAllWindows()"),
                                         qt.Qt.CTRL+qt.Qt.Key_Q)
                self.menuBar().insertItem('&File',self.menuFile)
                self.onInitMenuBar(self.menuBar())
            else:
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
            if QTVERSION < '4.0.0':
                self.menuTools= qt.QPopupMenu()
                self.menuTools.setCheckable(1)
                self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"), self.menuToolsAboutToShow)
                self.menuBar().insertItem("&Tools", self.menuTools)
            else:
                self.menuTools= qt.QMenu()
                #self.menuTools.setCheckable(1)
                self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"),
                             self.menuToolsAboutToShow)
                self.menuTools.setTitle("&Tools")
                self.menuBar().addMenu(self.menuTools)
                
        if self.options["MenuWindow"]:
            if QTVERSION < '4.0.0':
                self.menuWindow= qt.QPopupMenu()
                self.menuWindow.setCheckable(1)
                self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
                self.menuBar().insertItem("&Window", self.menuWindow)
            else:
                self.menuWindow= qt.QMenu()
                #self.menuWindow.setCheckable(1)
                self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
                self.menuWindow.setTitle("&Window")
                self.menuBar().addMenu(self.menuWindow)

        if self.options["MenuHelp"]:
            if QTVERSION < '4.0.0':
                self.menuHelp= qt.QPopupMenu(self)
                self.menuHelp.insertItem("&About", self.onAbout)
                self.menuHelp.insertItem("About &Qt",self.onAboutQt)
                self.menuBar().insertSeparator()
                self.menuBar().insertItem("&Help", self.menuHelp)
            else:
                self.menuHelp= qt.QMenu()
                self.menuHelp.addAction("&About", self.onAbout)
                self.menuHelp.addAction("About &Qt",self.onAboutQt)
                self.menuBar().addSeparator()
                self.menuHelp.setTitle("&Help")
                self.menuBar().addMenu(self.menuHelp)

    def menuWindowAboutToShow(self):
        if DEBUG:
            print "menuWindowAboutToShow"
        self.menuWindow.clear()
        if len(self.mdi.windowList())==0: return
        if QTVERSION < '4.0.0':
            self.menuWindow.insertItem("&Cascade", self.windowCascade)
            self.menuWindow.insertItem("&Tile", self.windowTile)
            self.menuWindow.insertItem("&Tile Horizontally", self.windowHorizontal)
            self.menuWindow.insertItem("&Tile Vertically", self.windowVertical)
            self.menuWindow.insertSeparator()
            num= 0
            self.menuWindowMap= {}
            for window in self.mdi.windowList():
                if QTVERSION < '4.0.0':
                    idx= self.menuWindow.insertItem("&%d %s"%(num, str(window.caption())),
                                                    self.menuWindowActivated)
                else:
                    idx= self.menuWindow.addAction("&%d %s"%(num, str(window.windowTitle())),
                                                    self.menuWindowActivated)                
                self.menuWindowMap[idx]= window
                num += 1
                if window==self.mdi.activeWindow():
                    if QTVERSION < '4.0.0':
                        self.menuWindow.setItemChecked(idx, 1)
                    else:
                        if DEBUG:
                            print "self.menuWindow.setItemChecked(idx, 1) equivalent missing"
        else:
            self.menuWindow.addAction("&Cascade", self.windowCascade)
            self.menuWindow.addAction("&Tile", self.windowTile)
            self.menuWindow.addAction("&Tile Horizontally", self.windowHorizontal)
            self.menuWindow.addAction("&Tile Vertically", self.windowVertical)
            windows=self.mdi.windowList()
            if len(windows) > 0: self.menuWindow.addSeparator()
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
            print "menuWindowActivated idx = ",idx
        if idx is None:return
        if self.menuWindowMap[idx].isHidden():
            self.menuWindowMap[idx].show()
            if QTVERSION < '4.0.0':
                self.menuWindowMap[idx].raiseW()
            else:
                self.menuWindowMap[idx].raise_()
        self.menuWindowMap[idx].setFocus()

    def __connectFollow(self):
            self.connect(self.mdi, qt.SIGNAL("windowActivated(QWidget*)"), self.onWindowActivated)

    def __disconnectFollow(self):
            self.disconnect(self.mdi, qt.SIGNAL("windowActivated(QWidget*)"), self.onWindowActivated)

    def setFollowActiveWindow(self, follow):
            if follow!=self.followActiveWindow:
                    if not follow: self.__disconnectFollow()
                    else: self.__connectFollow()
                    self.followActiveWindow= follow

    def onWindowActivated(self, win):
            print "Window activated"
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
            print "menuToolsAboutToShow"
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
            print "menuToolsActivated idx = ",idx    
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
    except getopt.error,msg:
        print msg
        sys.exit(1)
    # --- waiting widget
    if QTVERSION < '4.0.0':
        wa= qt.QMessageBox("PyMca", "PyMca v. 1.5 loading ...", qt.QMessageBox.NoIcon,
                                qt.QMessageBox.NoButton, qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton, None, None)
        wa.show()
    kw={}
    for opt, arg in opts:
        if  opt in ('--spec'):
            kw['spec'] = arg
        elif opt in ('--shm'):
            kw['shm']  = arg
    #demo = McaWindow.McaWidget(**kw)
    demo = PyMca()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        # --- close waiting widget
        wa.close()
        app.exec_loop()
    else:
        demo.show()
        app.exec_()

if __name__ == '__main__':
    main(sys.argv)

