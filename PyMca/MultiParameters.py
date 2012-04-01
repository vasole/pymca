#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
import sys
from PyMca import PyMcaQt as qt
from PyMca import Parameters
QTVERSION = qt.qVersion()
from PyMca import McaTable

DEBUG = 0

class ParametersTab(qt.QTabWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QTabWidget.__init__(self, parent, name, fl)
        else:
            qt.QTabWidget.__init__(self, parent)

        #if name == None:
        #    self.setName("FitParameters")
        
        #geometry
        #self.resize(570,300)
        #self.setCaption(self.trUtf8(name))
            
        #initialize the numer of tabs to 1
        #self.TabWidget=QTabWidget(self,"ParametersTab")
        #self.TabWidget.setGeometry(QRect(25,25,450,350))
        #the widgets in the notebook
        self.views={}
        #the names of the widgets (to have them in order)
        self.tabs=[]
        #the widgets/tables themselves
        self.tables={}
        self.mcatable=None
        if QTVERSION < '4.0.0':
            self.setMargin(10)
        else:
            if DEBUG: "self.setMargin(10) omitted"
        self.setview(name="Region 1")

    def setview(self,name=None,fitparameterslist=None):
        if name is None:
            name = self.current
        if name in self.tables.keys():
            table=self.tables[name]
        else:
            #create the parameters instance
            self.tables[name]=Parameters.Parameters(self)
            table=self.tables[name]
            self.tabs.append(name)
            self.views[name]=table
            #if qVersion() >= '3.0.0':
            #    self.addTab(table,self.trUtf8(name))
            #else:
            #    self.addTab(table,self.tr(name))
            self.addTab(table,str(name))
        if fitparameterslist is not None:
            table.fillfromfit(fitparameterslist)
        #print "SHowing page ",name
        if QTVERSION < '4.0.0':
            self.showPage(self.views[name])
        else:
            self.setCurrentWidget(self.views[name])
        self.current=name        

    def renameview(self,oldname=None,newname=None):
        error = 1
        if newname is not None:
            if newname not in self.views.keys():            
                if oldname in self.views.keys():
                    parameterlist=self.tables[oldname].fillfitfromtable()
                    self.setview(name=newname,fitparameterslist=parameterlist)
                    self.removeview(oldname)
                    error = 0
        return error

    def fillfromfit(self,fitparameterslist,current=None):
        if current is None:
            current=self.current
        #for view in self.tables.keys():
        #    self.removeview(view)
        self.setview(fitparameterslist=fitparameterslist,name=current)
        
    def fillfitfromtable(self,*vars,**kw):
        if len(vars) > 0:
            name=vars[0]
        elif 'view' in kw:
            name=kw['view']
        elif 'name' in kw:
            name=kw['name']
        else:
            name=self.current
        if hasattr(self.tables[name],'fillfitfromtable'):
            return self.tables[name].fillfitfromtable()
        else:
            return None       
    
    def removeview(self,*vars,**kw):
        error = 1
        if len(vars) > 0:
            view=vars[0]
        elif 'view' in kw:
            view=kw['view']
        elif 'name' in kw:
            view=kw['name']
        else:
            return error
        if view == self.current:
            return error
        if view in self.views.keys():
                self.tabs.remove(view)
                if QTVERSION < '4.0.0':
                    self.removePage(self.tables[view])
                    self.removePage(self.views[view])
                else:
                    index = self.indexOf(self.tables[view])
                    self.removeTab(index)
                    index = self.indexOf(self.views[view])
                    self.removeTab(index)
                del self.tables[view]
                del self.views[view]
                error =0
        return error

    def removeallviews(self,keep='Fit'):
        for view in list(self.tables.keys()):
            if view != keep:
                self.removeview(view)

    def fillfrommca(self,mcaresult):
        #for view in self.tables.keys():
        #    self.removeview(view)
        self.removeallviews()
        region = 0
        for result in mcaresult:
             #if result['chisq'] is not None:
                region=region+1
                self.fillfromfit(result['paramlist'],current='Region '+\
                                 "%d" % region)
        name='MCA'
        if name in self.tables:
           table=self.tables[name]
        else:
           self.tables[name]=McaTable.McaTable(self)
           table=self.tables[name]       
           self.tabs.append(name)
           self.views[name]=table
           #self.addTab(table,self.trUtf8(name))
           self.addTab(table,str(name))
           if QTVERSION < '4.0.0':
               self.connect(table,qt.PYSIGNAL('McaTableSignal'),self.__forward)
           else:
               self.connect(table,qt.SIGNAL('McaTableSignal'),self.__forward)
        table.fillfrommca(mcaresult)
        self.setview(name=name)        
        return
        
    def __forward(self,ddict):
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('MultiParametersSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('MultiParametersSignal'),(ddict))


    def gettext(self,**kw):
        if "name" in kw:
            name = kw["name"]
        else:
            name = self.current
        table = self.tables[name]
        lemon = ("#%x%x%x" % (255,250,205)).upper()
        if QTVERSION < '4.0.0':
            hb = table.horizontalHeader().paletteBackgroundColor()
            hcolor = ("#%x%x%x" % (hb.red(), hb.green(), hb.blue())).upper()
        else:
            if DEBUG:
                print("Actual color to ge got")
            hcolor = ("#%x%x%x" % (230,240,249)).upper()
        text=""
        text+=("<nobr>")
        text+=( "<table>")
        text+=( "<tr>")
        if QTVERSION < '4.0.0':
            ncols = table.numCols()
        else:
            ncols = table.columnCount()
        for l in range(ncols):
            text+=('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text+=(str(table.horizontalHeader().label(l)))
            else:
                text+=(str(table.horizontalHeaderItem(l).text()))
            text+=("</b></td>")
        text+=("</tr>")
        if QTVERSION < '4.0.0': nrows = table.numRows()
        else: nrows = table.rowCount()
        for r in range(nrows):
            text+=("<tr>")
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r,0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                color = "white"
                b="<b>"                
            else:
                b=""
                color = lemon
            try:
                #MyQTable item has color defined
                cc = table.item(r,0).color
                cc = ("#%x%x%x" % (cc.red(),cc.green(),cc.blue())).upper()
                color = cc
            except:
                pass
            for c in range(ncols):
                if QTVERSION < '4.0.0':
                    newtext = str(table.text(r,c))
                else:
                    item = table.item(r, c)
                    newtext = ""
                    if item is not None:
                        newtext = str(item.text())
                if len(newtext):
                    finalcolor = color
                else:
                    finalcolor = "white"
                if c<2:
                    text+=('<td align="left" bgcolor="%s">%s' % (finalcolor,b))
                else:
                    text+=('<td align="right" bgcolor="%s">%s' % (finalcolor,b))
                text+=(newtext)
                if len(b):
                    text+=("</td>")
                else:
                    text+=("</b></td>") 
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r,0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                text+=("</b>")
            text+=("</tr>")
            #text+=( str(qt.QString("<br>"))
            text+=("\n")
        text+=("</table>")
        text+=("</nobr>")
        return text

    def getHTMLText(self, **kw):
        return self.gettext(**kw)

    if QTVERSION > '4.0.0':
        def getText(self, **kw):
            if "name" in kw:
                name = kw["name"]
            else:
                name = self.current
            table = self.tables[name]
            text=""
            if QTVERSION < '4.0.0':
                ncols = table.numCols()
            else:
                ncols = table.columnCount()
            for l in range(ncols):
                if QTVERSION < '4.0.0':
                    text+=(str(table.horizontalHeader().label(l)))
                else:
                    text+=(str(table.horizontalHeaderItem(l).text()))+"\t"
            text+=("\n")
            if QTVERSION < '4.0.0':
                nrows = table.numRows()
            else:
                nrows = table.rowCount()
            for r in range(nrows):
                if QTVERSION < '4.0.0':
                    newtext = str(table.text(r,0))
                else:
                    item = table.item(r, 0)
                    newtext = ""
                    if item is not None:
                        newtext = str(item.text())+"\t"
                for c in range(ncols):
                    if QTVERSION < '4.0.0':
                        newtext = str(table.text(r,c))
                    else:
                        newtext = ""                        
                        if c != 4:
                            item = table.item(r, c)
                            if item is not None:
                                newtext = str(item.text())
                        else:
                            item = table.cellWidget(r, c)
                            if item is not None:
                                newtext = str(item.currentText())
                    text+=(newtext)+"\t"
                text+=("\n")
            text+=("\n")
            return text

def test():
    a = qt.QApplication(sys.argv)
    qt.QObject.connect(a,qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))
    w = ParametersTab()
    if QTVERSION < '4.0.0':a.setMainWidget(w)
    w.show()
    if 1:
        import specfile
        import Specfit
        from numpy.oldnumeric import array,Float,concatenate,arange
        sf=specfile.Specfile('02021201.dat')
        scan=sf.select('14')
        #sf=specfile.Specfile('02022101.dat')
        #scan=sf.select('11')
        mcadata=scan.mca(1)
        y=array(mcadata)
        #x=arange(len(y))
        x=arange(len(y))*0.0200511-0.003186
        fit=Specfit.Specfit()
        fit.setdata(x=x,y=y)
        fit.importfun("SpecfitFunctions.py")
        fit.settheory('Hypermet')
        fit.configure(Yscaling=1.,
                      WeightFlag=1,
                      PosFwhmFlag=1,
                      HeightAreaFlag=1,
                      FwhmPoints=50,
                      PositionFlag=1,
                      HypermetTails=1)        
        fit.setbackground('Linear')
        if 0:
            # build a spectrum array
            f=open("spec.arsp",'r')
            #read the spectrum datas
            x=array([],Float)
            y=array([],Float)
            tmp=f.readline()[:-1]
            while (tmp != ""):
                tmpSeq=tmp.split()
                x=concatenate((x,[float(tmpSeq[0])]))
                y=concatenate((y,[float(tmpSeq[1])]))
                tmp=f.readline()[:-1]
            fit.setdata(x=x,y=y)
        if 1:
            mcaresult=fit.mcafit(x=x,xmin=x[70],xmax=x[500])
            w.fillfrommca(mcaresult)
        else:
            fit.estimate()
            fit.startfit()
            w.fillfromfit(fit.paramlist,current='Fit')
            w.removeview(view='Region 1')
    if QTVERSION < '4.0.0':
        a.exec_loop()
    else:
        a.exec_()
        
if __name__ == "__main__":
    bench=0
    if bench:
        import pstats
        import profile
        profile.run('test()',"test")
        p=pstats.Stats("test")
        p.strip_dirs().sort_stats(-1).print_stats()
    else:
        test()
