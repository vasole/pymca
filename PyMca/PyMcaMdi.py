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
import qt
#import McaWindow
from PyMca_Icons import IconDict
from PyMca_help  import HelpDict
DEBUG = 0

__version__ = "1.5"

class PyMca(qt.QMainWindow):
    def __init__(self, parent=None, name="PyMca Mdi", fl=qt.Qt.WDestructiveClose, options={}):
        qt.QMainWindow.__init__(self, parent, name, fl)
        self.setCaption(name)

        self.options= {}
        self.options["FileToolBar"]= options.get("FileToolBar", 1)
        self.options["WinToolBar"] = options.get("WinToolBar", 1)
        self.options["MenuFile"]   = options.get("MenuFile", 1)
        self.options["MenuTools"]  = options.get("MenuTools", 1)
        self.options["MenuWindow"] = options.get("MenuWindow", 1)
        self.options["MenuHelp"]   = options.get("MenuHelp", 1)

        self.splitter = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Horizontal)
        self.printer= qt.QPrinter()
        if qt.qVersion()>="3.0.0":
                self.mdi= qt.QWorkspace(self.splitter)
                self.mdi.setScrollBarsEnabled(1)
                #self.setCentralWidget(self.mdi)
        else:
                self.mdi= qt.QWorkspace(self.splitter)
                self.mdi.setBackgroundColor(qt.QColor("gainsboro"))
                #self.setCentralWidget(self.mdi)
        self.setCentralWidget(self.splitter)
        self.splitter.moveToLast(self.mdi)

        #self.setDockEnabled(qt.Qt.DockTop, 0)

        self.initIcons()
        self.initMenuBar()
        self.initToolBar()

        self.followActiveWindow= 0

        self.mdi.show()
        #self.resize(600,400)

    def initIcons(self):
        self.Icons= {}
        for (name, icon) in IconDict.items():
            pixmap= qt.QPixmap(icon)
            self.Icons[name]= qt.QIconSet(pixmap)

    def initToolBar(self):
        if self.options["FileToolBar"]:
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

        self.onInitToolBar()

        if self.options["WinToolBar"]:
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
                    window.parentWidget().raiseW()
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
                    window.parentWidget().raiseW()
                    i+=1
            self.mdi.update()
            self.update()
            #if self.followActiveWindow: self.__connectFollow()

    def windowFullScreen(self):
            if len(self.mdi.windowList()):
                    self.mdi.activeWindow().showMaximized()


    def initMenuBar(self):
        if self.options["MenuFile"]:
            self.menuFile= qt.QPopupMenu(self.menuBar())
            idx= self.menuFile.insertItem(self.Icons["fileopen"], qt.QString("&Open"), self.onOpen, qt.Qt.CTRL+qt.Qt.Key_O)
            self.menuFile.setWhatsThis(idx, HelpDict["fileopen"])
            idx= self.menuFile.insertItem(self.Icons["filesave"], "&Save", self.onSave, qt.Qt.CTRL+qt.Qt.Key_S)
            self.menuFile.setWhatsThis(idx, HelpDict["filesave"])
            self.menuFile.insertItem("Save &as", self.onSaveAs)
            self.menuFile.insertSeparator()
            idx= self.menuFile.insertItem(self.Icons["fileprint"], "&Print", self.onPrint, qt.Qt.CTRL+qt.Qt.Key_P)
            self.menuFile.setWhatsThis(idx, HelpDict["fileprint"])
            self.menuFile.insertSeparator()
            self.menuFile.insertItem("&Quit", qt.qApp, qt.SLOT("closeAllWindows()"), qt.Qt.CTRL+qt.Qt.Key_Q)
            self.menuBar().insertItem('&File',self.menuFile)

        self.onInitMenuBar(self.menuBar())

        if self.options["MenuTools"]:
            self.menuTools= qt.QPopupMenu()
            self.menuTools.setCheckable(1)
            self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"), self.menuToolsAboutToShow)
            self.menuBar().insertItem("&Tools", self.menuTools)

        if self.options["MenuWindow"]:
            self.menuWindow= qt.QPopupMenu()
            self.menuWindow.setCheckable(1)
            self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
            self.menuBar().insertItem("&Window", self.menuWindow)

        if self.options["MenuHelp"]:
            self.menuHelp= qt.QPopupMenu(self)
            self.menuHelp.insertItem("&About", self.onAbout)
            self.menuHelp.insertItem("About &Qt",self.onAboutQt)
            self.menuBar().insertSeparator()
            self.menuBar().insertItem("&Help", self.menuHelp)

    def menuWindowAboutToShow(self):
        if DEBUG:
            print "menuWindowAboutToShow"
        self.menuWindow.clear()
        if len(self.mdi.windowList())==0: return
        self.menuWindow.insertItem("&Cascade", self.windowCascade)
        self.menuWindow.insertItem("&Tile", self.windowTile)
        self.menuWindow.insertItem("&Tile Horizontally", self.windowHorizontal)
        self.menuWindow.insertItem("&Tile Vertically", self.windowVertical)
        self.menuWindow.insertSeparator()

        num= 0
        self.menuWindowMap= {}
        for window in self.mdi.windowList():
                idx= self.menuWindow.insertItem("&%d %s"%(num, str(window.caption())), self.menuWindowActivated)
                self.menuWindowMap[idx]= window
                num += 1
                if window==self.mdi.activeWindow():
                        self.menuWindow.setItemChecked(idx, 1)

    def menuWindowActivated(self, idx):
        if DEBUG:
            print "menuWindowActivated idx = ",idx
        if self.menuWindowMap[idx].isHidden():
           self.menuWindowMap[idx].show()
           self.menuWindowMap[idx].raiseW()
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
    app.setMainWidget(demo)
    demo.show()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
    # --- close waiting widget
    wa.close()
    app.exec_loop()

if __name__ == '__main__':
    main(sys.argv)

