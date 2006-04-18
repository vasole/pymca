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
from Tkinter import *
from Tkinter import _cnfmerge
from BasicMenu import *
import os
try:
    HOME=os.getenv('HOME')
except:
    HOME=None
if HOME is not None:
    os.environ['HOME'] = HOME    
else:
    os.environ['HOME'] = "."

SPECFITDEFAULTS={'FileAction':0,
                 'infile':os.environ['HOME']+'/.specfitdefaults.py',
                 'outfile':os.environ['HOME']+'/.specfitdefaults.py',
                 'Geometry':"600x450+50+50",
                 'NoConstrainsFlag':0,
                 'BackgroundIndex':1,
                 'TheoryIndex':0,
                 'PosFwhmFlag':1,
                 'HeightAreaFlag':0,
                 'SameFwhmFlag':1,
                 'PositionFlag':0,
                 'EtaFlag':0,
                 'WeightFlag':0,
                 'Yscaling':1.0,
                 'Xscaling':1.0,
                 'FwhmPoints':10,
                 'Sensitivity':3.25,
                 'McaMode':0}

COMMAND   = 1
SEPARATOR = 2
CASCADE   = 3

DEBUG = 0

class SpecfitConf(BasicMenu):
    def __init__(self,master,fitconfig=None,fitdefaults=None,
                configfile=None,cnf={},**kw):
        """
        This module deals with the configuration of Specfit.
        1.- Implements a configuration Menu.
        2.- The dictionary fitconfig contains the current configuration.
        3.- The dictionary fitdefaults contains the default configuration.
        """
        if DEBUG:
            print 'BasicMenu : init'

        cnf = _cnfmerge((cnf,kw))
        BasicMenu.__init__(self,master,cnf)

        if configfile is not None:
            self.configfile = configfile
        else:
            self.configfile = os.environ['HOME']+'/.specfitdefaults.py'
        
        if fitdefaults is not None:
            self.fitdefaults = fitdefaults
        else:
            self.fitdefaults = {}
            self.loaddefaults()

        if fitconfig is not None:
            self.fitconfig = fitconfig
        else:
            self.fitconfig = self.fitdefaults.copy()            
            
        self.theory_list = ['Background & Gaussians',
                           'Background & Lorentz',
                           'Back. Area Gaussians',
                           'Back. Area Lorentz',
                           'Pseudo-Voigt Line',
                           'Area Pseudo-Voigt',
                           'Back. & Step Down',
                           'Back. & Step Up',
                           'Back. & Slit']

        config_menu  = ["Config", None,
                          [COMMAND, "Configure",  self.askconfig],
                          [SEPARATOR, ],
                          [COMMAND, "Load Config",  self._loadconfig],
                          [COMMAND, "Save Config",  self._saveconfig],
                          [SEPARATOR,],
                          [COMMAND, "Load Default",  self.setdefaults],
                          [COMMAND, "Save As Default", self.savedefaults],
                          [SEPARATOR, ],
                          [COMMAND, "Exit",       None]   ]

        self.create_menu(config_menu,side='left')
        self.fitconfigupdated = IntVar()

    def askconfig(self):
        sheet1=Sheet(notetitle='Fit Theory',
            help='Select default built-in\nfit function',
            fields=(TitleField('Fit Function'),
                RadioField('TheoryIndex',
                values=self.theory_list)
                ))
        sheet2=Sheet(notetitle='Background',
            help='Select background function',
            fields=(TitleField('Default Background Function'),
                RadioField('BackgroundIndex',
                values=['Constant','Linear','Exponential','Internal',
                'No Background'])
                ))
        sheet3=Sheet(notetitle='Constraints',
            help='Set initial constraints\nfor built in functions',
            fields=(TitleField('Default Constraints'),
                CheckField('HeightAreaFlag','Force positive Height/Area'),
                CheckField('PositionFlag','Force position in interval',
                            help='Prevent peak positions outside\nthe fitting range'),
                CheckField('PosFwhmFlag','Force positive FWHM'),
                CheckField('SameFwhmFlag','All peaks same FWHM',
                            help='FWHM of peaks set equal\n to that of the largest one'),
                CheckField('EtaFlag','Eta between 0.0 and 1.0',
                            help='Force 0.0 < pseudo-Voigt parameter eta < 1.0'),
                CheckField('NoConstrainsFlag','No Constrains',
                            help='All other flags are ignored\nif enabled'),
                ))
        sheet4=Sheet(notetitle='Weight & Search',
            help='Adjust weight and\n search parameters',
            fields=(TitleField('Default Weight'),
                RadioField('WeightFlag',values=['No Weight','Statistical']),
                TitleField('Peak Search'),
                EntryField('FwhmPoints', 'FWHM Points',
                           help='Estimated FWHM in points'),
                EntryField('Sensitivity','Sensitivity',
                           help='A positive number around 3'),
                EntryField('Yscaling', 'Y factor',
                           help='Factor to apply to Y\nprior to peak search'),
                CheckField('McaMode', 'MCA Mode',
                           help='Not yet used')
                ))
        dummy = {}
        for key in self.fitconfig.keys():
            if key != '__builtins__':
                dummy[key] = self.fitconfig[key]

        app=SheetDialog(self.master,title='Specfit Config',
                    sheets=(sheet1,sheet2,sheet3,sheet4),
                    type='notebook',
                    validate=0,
                    init=dummy,
                    default=dummy)
        if app.result is not None:
            for key in app.result.keys():
                if key != '__builtins__':
                    if (key != 'Xscaling') and (key != 'Yscaling'):
                        self.fitconfig[key] = app.result[key]
                    else:
                        try:
                            self.fitconfig[key] = float(app.result[key])
                        except:
                            tkMessageBox.showerror('Error',
                                "%s must be a number" % key)
            self.fitconfigupdated.set(1)
        del app
        
    def _loadconfig(self):
        dummy={}
        if self.fitconfig.has_key('infile'):
            dummy['infile']=self.fitconfig['infile']
        elif self.fitdefaults.has_key('infile'):
            dummy['infile']=self.fitdefaults['infile']
        sheet1= Sheet(notetitle='Load Config',
                fields=(TitleField('Configuration File'),   
                    FileOutput('infile','Output File')
                    ))
        sheet2= Sheet(notetitle='Dummy Config',
                fields=(TitleField('Configuration File'),   
                    FileOutput('infile','Output File')
                    ))
        app=SheetDialog(self.master,title='Specfit Config',
                    sheets=[sheet1],
                    type='notebook',
                    validate=0,
                    init=dummy,
                    default=dummy)
        if app.result is not None:
               file=app.result['infile']
               self.loadconfig(file) 
        del app
        
    def loadconfig(self,file=None):
        if DEBUG:
            print "Load config tries to read %s \n" % file
        dummy = {}
        if file is not None:
            try:
                execfile(file,dummy)
            except:
                tkMessageBox.showerror('Error',
                    "Error reading file %s" % file)
                return
            if DEBUG:
                print "dummy keys = ",dummy.keys()
            for key in dummy.keys():
                if key != '__builtins__':
                    self.fitconfig[key] = dummy[key]
                    if DEBUG: 
                        print key," =  ",self.fitconfig[key]
        del dummy
        self.fitconfigupdated.set(1)

    def _saveconfig(self):
        dummy={}
        if self.fitconfig.has_key('outfile'):
            dummy['outfile']=self.fitdefaults['outfile']
        sheet1= Sheet(notetitle='Save Config',
                fields=(TitleField('Configuration File'),   
                    FileOutput('outfile','Output File')
                    ))
        app=SheetDialog(self.master,title='Specfit Config',
                    sheets=[sheet1],
                    type='notebook',
                    validate=0,
                    init=dummy,
                    default=dummy)
        if app.result is not None:
               filename=app.result['outfile']
               self.fitconfig['outfile'] = filename
               self.saveconfig(filename) 
        del app

    def saveconfig(self,filen=None):
        if filen is not None:
            try:
                f=open(filen,'w')
            except:
                tkMessageBox.showerror('Error',
                    "Error opening output file %s" % filen)
                f.close()
                return
            self.fitconfig['Geometry']=self.master.geometry()
            for key in self.fitconfig.keys():
                if key != '__builtins__':
                    if type(self.fitconfig[key]) == type("string"):
                        f.write("%s='%s'\n" % (key,self.fitconfig[key]))
                    else:
                        f.write("%s=%s\n" % (key,self.fitconfig[key]))
                    if DEBUG:
                        print "saved ",key," = ",self.fitconfig[key]
            f.close()

    def loaddefaults(self):
        filename=os.environ['HOME']+'/.specfitdefaults.py'
        dummy = {}
        try:
            execfile(filename,dummy)
        except:
            #stay with the internal configuration
            dummy = SPECFITDEFAULTS.copy()
            tkMessageBox.showerror('Error',
                    "Error reading file %s" % filename)
        for key in SPECFITDEFAULTS.keys():
            self.fitdefaults[key]= SPECFITDEFAULTS[key] 
        for key in dummy.keys():
            if key != '__builtins__':
                self.fitdefaults[key] = dummy[key]
                if DEBUG:
                    print "loaded ",key," = ",self.fitdefaults[key]

    def setdefaults(self):
        """
        Tries to load the default configuration from the default file and
        sets the current configuration equal to the default configuration.
        """
        self.loaddefaults()
        for key in self.fitdefaults.keys():
            if key != '__builtins__':
                self.fitconfig[key] = self.fitdefaults[key]
        self.fitconfigupdated.set(1)
                

    def savedefaults(self):
        filename=os.environ['HOME']+'/.specfitdefaults.py'
        self.saveconfig(filen=filename)

        
if __name__ == '__main__':
    root = Tk()
    menu=SpecfitConf(root)
    item = 10
    def test(item='default'):
        print "Hello from test function"
        print "Received argument = ",item
        testdialog()
    #config_menu  = [  "Config", None,
    #                      [COMMAND, "Configure",  lambda i=item:test(i)],
    #                      [SEPARATOR, ],
    #                      [COMMAND, "Load Config",  None],
    #                      [COMMAND, "Save Config",  None],
    #                      [SEPARATOR,],
    #                      [COMMAND, "Load Defaults",  None],
    #                      [COMMAND, "Save Defaults", None],
    #                      [SEPARATOR, ],
    #                      [COMMAND, "Exit",       None]   ]
    #menu.create_menu(config_menu)
    #help_menu = [ "Help",None,
    #                    [COMMAND, "Configure",  lambda i=20:test(i)]]
    #menu.update_menu(help_menu)

    menu.pack()
    root.mainloop()
