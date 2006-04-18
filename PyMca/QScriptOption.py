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
import sys
from qt import *
import TextField
#import RadioField
import CheckField
import EntryField
import TabSheets

TupleType=type(())

def uic_load_pixmap_RadioField(name):
    pix = QPixmap()
    m = QMimeSourceFactory.defaultFactory().data(name)

    if m:
        QImageDrag.decode(m,pix)

    return pix

class QScriptOption(TabSheets.TabSheets):
    def __init__(self,parent = None,name=None,modal=1,fl = 0,
                sheets=(),default=None,nohelp=1,nodefaults=1):
        TabSheets.TabSheets.__init__(self,parent,name,modal,fl,
                                    nohelp,nodefaults)
        if name is not None:
            self.setCaption(str(name))
        self.sheets={}
        self.sheetslist=[]
        self.default=default
        self.output={}
        self.output.update(self.default)
        if qVersion() >= '3.0.0':
            ntabs=self.tabWidget.count()
        else:
            ntabs = 2

        for sheet in sheets:
            name=sheet['notetitle']
            a=FieldSheet(fields=sheet['fields'])
            self.sheets[name]=a
            a.setdefaults(self.default)
            self.sheetslist.append(name)
            self.tabWidget.addTab(self.sheets[name],str(name))
            self.tabWidget.showPage(self.sheets[name])
        #remove anything not having to do with my sheets
        for i in range(ntabs):
            if qVersion() >= '3.0.0':
                page=self.tabWidget.page(0)
                self.tabWidget.removePage(page)
            else:
                self.tabWidget.setCurrentPage(i)
                self.tabWidget.removePage(self.tabWidget.currentPage())
            
        #perform the binding to the buttons
        self.connect(self.buttonOk,SIGNAL("clicked()"),self.myaccept)
        self.connect(self.buttonCancel,SIGNAL("clicked()"),self.myreject)
        if not nodefaults:
            self.connect(self.buttonDefaults,SIGNAL("clicked()"),self.defaults)
        if not nohelp:
            self.connect(self.buttonHelp,SIGNAL("clicked()"),self.myhelp)
        
        
    def myaccept(self):
        self.output.update(self.default)
        for name,sheet in self.sheets.items():
            self.output.update(sheet.get())
        #avoid pathologicval None cases
        for key in self.output.keys():
            if self.output[key] is None:
                if self.default.has_key(key):
                    self.output[key]=self.default[key]    
        
        self.accept()
        return
               
    def myreject(self):
        self.output={}
        self.output.update(self.default)
        self.reject()
        return

    def defaults(self):
        self.output={}
        self.output.update(self.default)
        
    def myhelp(self):    
        print "Default - Sets back to the initial parameters"
        print "Cancel  - Sets back to the initial parameters and quits" 
        print "OK      - Updates the parameters and quits" 
        
class FieldSheet(QWidget):
    def __init__(self,parent = None,name=None,fl = 0,fields=()):
        QWidget.__init__(self,parent,name,fl)
        layout= QVBoxLayout(self)
        layout.setAutoAdd(1)
        #self.fields = ([,,,])
        self.fields=[]
        self.nbfield= 1
        for field in fields:
            fieldtype=field[0]
            if len(field) == 3:
                key = field[1]
            else:
                key = None
            parameters=field[-1]
            if fieldtype == "TextField":
                self.fields.append(MyTextField(self,keys=key,params=parameters))
            if fieldtype == "CheckField":
                self.fields.append(MyCheckField(self,keys=key,params=parameters))
            if fieldtype == "EntryField":
                self.fields.append(MyEntryField(self,keys=key,params=parameters))
            if fieldtype == "RadioField":
                self.fields.append(RadioField(self,keys=key,params=parameters))
            
    def get(self):
        result={}
        for field in self.fields:
            result.update(field.getvalue())
        return result

    def setdefaults(self,dict):
        for field in self.fields:
            field.setdefaults(dict)
        return

class MyTextField(TextField.TextField):
    def __init__(self,parent = None,name = None,fl = 0,
                    keys=(), params = ()):
        TextField.TextField.__init__(self,parent,name,fl)
        self.TextLabel.setText(str(params))
                 
    def getvalue(self):
        pass
        return
            
    def setvalue(self):
        pass    
        return

    def setdefaults(self,dict):
        pass
        return

class MyEntryField(EntryField.EntryField):
    def __init__(self,parent = None,name = None,fl = 0,
                    keys=(), params = ()):
        EntryField.EntryField.__init__(self,parent,name,fl)
        self.dict={}
        if type(keys) == TupleType:
            for key in keys:
                self.dict[key]=None
        else:
            self.dict[keys]=None
        self.TextLabel.setText(str(params))
        self.connect(self.Entry,SIGNAL("textChanged(const QString&)"),self.setvalue)
                 
    def getvalue(self):
        return self.dict
            
    def setvalue(self,value):
        for key in self.dict.keys():
            self.dict[key]=str(value)
        return

    def setdefaults(self,dict):
        for key in self.dict.keys():
            if dict.has_key(key):
                self.dict[key]=dict[key]
                self.Entry.setText(str(dict[key])) 
        return



class MyCheckField(CheckField.CheckField):
    def __init__(self,parent = None,name = None,fl = 0,
                    keys=(), params = ()):
        CheckField.CheckField.__init__(self,parent,name,fl)
        self.dict={}
        if type(keys) == TupleType:
            for key in keys:
                    self.dict[key]=None
        else:
            self.dict[keys]=None
        self.CheckBox.setText(str(params))
        self.connect(self.CheckBox,SIGNAL("stateChanged(int)"),self.setvalue)
                 
    def getvalue(self):
        return self.dict
            
    def setvalue(self,value):
        if value:
            val=1
        else:
            val=0                
        for key in self.dict.keys():
            self.dict[key]=val
        return

    def setdefaults(self,dict):
        for key in self.dict.keys():
            if dict.has_key(key):
                if int(dict[key]):
                    self.CheckBox.setChecked(1)
                    self.dict[key]=1
                else:
                    self.CheckBox.setChecked(0)
                    self.dict[key]=0 
        return

class RadioField(QWidget):
    def __init__(self,parent = None,name = None,fl = 0,
                            keys=(), params = ()):
            QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("RadioField")

            #self.resize(166,607)
            self.setSizePolicy(QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
            self.setCaption(str("RadioField"))

            RadioFieldLayout = QHBoxLayout(self,11,6,"RadioFieldLayout")

            self.RadioFieldBox = QButtonGroup(self,"RadioFieldBox")
            self.RadioFieldBox.setSizePolicy(QSizePolicy(1,1,0,0,self.RadioFieldBox.sizePolicy().hasHeightForWidth()))
            self.RadioFieldBox.setTitle(str(""))
            self.RadioFieldBox.setColumnLayout(0,Qt.Vertical)
            self.RadioFieldBox.layout().setSpacing(6)
            self.RadioFieldBox.layout().setMargin(11)
            RadioFieldBoxLayout = QVBoxLayout(self.RadioFieldBox.layout())
            RadioFieldBoxLayout.setAlignment(Qt.AlignTop)
            Layout1 = QVBoxLayout(None,0,6,"Layout1")
            
            self.dict={}
            if type(keys) == TupleType:
                for key in keys:
                    self.dict[key]=1
            else:
                self.dict[keys]=1
            self.RadioButton=[]
            i=0
            for text in params:
                self.RadioButton.append(QRadioButton(self.RadioFieldBox,
                                                        "RadioButton"+`i`))
                self.RadioButton[-1].setSizePolicy(QSizePolicy(1,1,0,0,
                                self.RadioButton[-1].sizePolicy().hasHeightForWidth()))
                self.RadioButton[-1].setText(str(text))
                Layout1.addWidget(self.RadioButton[-1])
                i=i+1
            
            RadioFieldBoxLayout.addLayout(Layout1)
            RadioFieldLayout.addWidget(self.RadioFieldBox)
            self.RadioButton[0].setChecked(1)
            self.connect(self.RadioFieldBox,SIGNAL("clicked(int)"),
                                self.setvalue)
                 
    def getvalue(self):
        return self.dict
            
    def setvalue(self,value):
        if value:
            val=1
        else:
            val=0                
        for key in self.dict.keys():
            self.dict[key]=val
        return

    def setdefaults(self,dict):
        for key in self.dict.keys():
            if dict.has_key(key):
                self.dict[key]=dict[key]
                i=int(dict[key])
                self.RadioButton[i].setChecked(1)
        return

def test():
    a = QApplication(sys.argv)
    QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"))
    #w = FieldSheet(fields=(["TextField",'Simple Entry'],
    #                       ["EntryField",'entry','MyLabel'],
    #                       ["CheckField",'label','Check Label'],
    #                       ["RadioField",'radio',('Button1','hmmm','3')]))
    sheet1={'notetitle':"First Sheet",
            'fields':(["TextField",'Simple Entry'],
                           ["EntryField",'entry','MyLabel'],
                           ["CheckField",'label','Check Label'])}
    sheet2={'notetitle':"Second Sheet",
            'fields':(["TextField",'Simple Radio Buttons'],
                           ["RadioField",'radio',('Button1','hmmm','3')])}
    w=QScriptOption(name='QScriptOptions',sheets=(sheet1,sheet2),
                            default={'radio':1,'entry':'type here','label':1})
    a.setMainWidget(w)
    w.show()
    a.exec_loop()
    print w.output
    
if __name__ == "__main__":
    test()
