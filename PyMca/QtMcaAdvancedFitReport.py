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
import os
import sys
MATPLOTLIB = False
try:
    #for the time being I force to have Qt.
    #This is to use matplotlib on Qt4 and Qwt on Qt3 and Qt2.
    #If matplotlib is installed this module should be able
    #to generate the HTML from the fitresult file without having Qt
    #installed. To test just comment next line.
    import PyQt4.Qt as qt
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        MATPLOTLIB = True
    except:
        import QtBlissGraph
except:
    import qt
    if qt.qVersion() < '3.0.0':
        try:
            from matplotlib.font_manager import FontProperties
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure
            MATPLOTLIB = True
        except:
            import QtBlissGraph
    else:        
        import QtBlissGraph
import ConfigDict
import time
import string
import PyMcaLogo

class QtMcaAdvancedFitReport:
    def __init__(self, fitfile = None, outfile = None, outdir = None,
                    sourcename = None,
                    selection = None,
                    fitresult = None,htmltext=None,
                    concentrations=None, table = None):
        
        self.concentrations = concentrations
        if table is None: table = 2
        self.tableFlag = table
        if fitfile is not None:
            #generate output from fit result file
            self.fitfile = fitfile
            self.outfile = outfile
            self.outdir  = outdir
            self.generateReportFromFitFile()
        else:
            #generate output from fitresult INCLUDING fit file
            self.fitfile = fitfile
            self.outfile = outfile
            self.outdir  = outdir
            self.sourcename=sourcename
            self.selection =selection
            self.fitresult =fitresult
            if self.outfile is None:
                if selection is not None:
                    self.outfile = selection
            if (self.outfile is None) or (self.outfile == 'Unknown Origin'):
                if sourcename is not None:
                    self.outfile = os.path.basename(sourcename)
        self.outfile = string.replace(self.outfile," ","_")
        self.graph = None            
        if htmltext is None:
            htmltext={}
        self.otherhtmltext=htmltext
    
    def writeReport(self,text=None):
        if len(self.outfile) > 5:
            if self.outfile[-5:] != ".html":
                outfile = os.path.join(self.outdir, self.outfile+".html")
            else:
                outfile = os.path.join(self.outdir, self.outfile)
        else:
            outfile = os.path.join(self.outdir, self.outfile+".html")
        try:
            os.remove(outfile)
        except:
            pass
        concentrationsfile = outfile[:-5]+"_concentrations.txt"
        try:
            os.remove(concentrationsfile)
        except:
            pass
        if text is None:
            text = self.getText()
        f=open(outfile,"w")
        f.write(text)
        f.close()
        if len(self._concentrationsTextASCII) > 1:
             f=open(concentrationsfile, "w")
             f.write(self._concentrationsTextASCII)
             f.close()   
        return outfile

    def generateReportFromFitFile(self):
        d=ConfigDict.ConfigDict()
        d.read(self.fitfile)
        sourcename = "Unknown Source"
        selection  = "Unknown Selection"
        if d.has_key('info'):
            if d['info'].has_key('key'):selection=d['info']['key']
            elif d['info'].has_key('Key'):selection=d['info']['Key']            
            for key in d['info'].keys():
                if string.upper(key) == 'SOURCENAME':
                    sourcename = d['info'][key]
                elif (string.upper(key) == 'SELECTION') or (string.upper(key) == 'LEGEND'):
                    selection = d['info'][key]
        self.sourcename = sourcename
        self.selection  = selection
        if self.outfile is None:
            if  self.outdir is None:
                self.outdir = os.getcwd()
            self.outfile= os.path.basename(self.fitfile)
        else:
            if self.outdir is None:
                self.outdir = os.path.dirname(self.outfile)
            self.outfile= os.path.basename(self.outfile)
        if self.outdir == '':self.outdir = "."
        self.fitresult=d
        if d.has_key('concentrations'):
            self.concentrations = d['concentrations']
    
    def getText(self):
        newlinks = []
        for key in self.otherhtmltext.keys():
            newlinks.append(["#%s" % (key),"%s" % key])
        text =self.getHeader(newlinks)
        text+=self.getInfo()
        text+=self.getImage()
        text+=self.getParam()
        text+=self.getConcentrations()
        self._concentrationsTextASCII = self.getConcentrationsASCII()
        text+=self.getResult()            
        for key in self.otherhtmltext.keys():
             text+="\n"
             text+= "<H2><a NAME=""%s""></a><FONT color=#009999>" % key
             text+= "%s:" % key
             text+= "</FONT></H2>"
             text+= self.otherhtmltext[key]
             text+="<br>"
        text+=self.getFooter()
        return text
       
    def getHeader(self,addlink=None):
        link = [ ['http://www.esrf.fr', 'ESRF home'],
                 ['http://www.esrf.fr/computing/bliss/', 'BLISS home']]
        if self.concentrations is not None:
            link.append(['#Concentrations', 'Concentrations'])
        if self.tableFlag:link.append(['#Fit_Peak_Results', 'Fit Peak Results'])
        if addlink is not None:
            for item in addlink:
                link.append(item)       
        text =""
        text+= "<HTML>"
        text+= "<HEAD>"
        text+= "<TITLE>PyMCA : Advanced Fit Results</TITLE>"
        text+= "</HEAD>"
        text+= "<BODY TEXT=#000000 BGCOLOR=#FFFFFF ALINK=#ff6600 LINK=#0000cc VLINK=#0000cc marginwidth=10 marginheight=10  topmargin=10 leftmargin=10>"
        text+= "<CENTER>"
        text+= "<TABLE WIDTH=100%% border=0 Height=70>"
        text+= "  <TR>"
        text+= "    <TD><Font Size=5 Color=#0000cc>"
        text+= "        <b>PyMCA : Advanced Fit Results</b></Font>"
        text+= "    </td>"
        text+= "    <td rowspan=2 ALIGN=RIGHT VALIGN=bottom>"
        text+= "        <a HREF=""http://www.esrf.fr/"">"
        logofile = self.outdir + "/" + "PyMcaLogo.png"
        if not os.path.exists(logofile):
            pixmap = qt.QPixmap(PyMcaLogo.PyMcaLogo)
            pixmap.save(logofile,"PNG")            
        text+= "        <img SRC=%s ALT=""ESRF home"" WIDTH=55 HEIGHT=68 BORDER=0></a>" % "PyMcaLogo.png"
        text+= "    </td>"
        text+= "  </tr>"
        text+= "  <tr>"
        text+= "     <td width=100%%  VALIGN=bottom>"
        text+= "        <TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0 WIDTH=100%%>"
        text+= "          <TR>"
        text+= "            <TD WIDTH=100%% BGCOLOR=#ee22aa HEIGHT=17  ALIGN=LEFT VALIGN=middle>"
        text+= "            <FONT color=#000000>&nbsp;"
        for name in link:
            text+= "|&nbsp;&nbsp;<A STYLE=""color: #FFFFFF"" HREF=""%s"">%s</a>&nbsp;&nbsp;"%(tuple(name))
        text+= "            </FONT>"
        text+= "            </TD>"
        text+= "          </TR>"
        text+= "        </TABLE>"
        text+= "     </td>"
        text+= "  </tr>"
        text+= "  <tr>"
        text+= "     <td colspan=2 height=5><spacer type=block height=10 width=0>"
        text+= "     </td>"
        text+= "  </tr>"
        text+= "</table>"
        text+= "</center>"
        return text

    def getInfo(self):
        text =""
        text+= "<nobr><H2><FONT color=#0000cc>"
        text+= "Computed File :&nbsp;"
        text+= "</FONT>"
        text+= "<FONT color=#000000>"
        if self.fitfile is not None:
            if os.path.basename(self.fitfile) == self.fitfile:
                text+= "<b><I>%s</I></b>" % (os.getcwd()+"/"+self.fitfile)
            else:
                text+= "<b><I>%s</I></b>" % (self.fitfile)
        else:
            text+= "<b><I>%s</I></b>" % (self.outdir+"/"+self.outfile+".fit")
            #and I have to generate it!!!!!!!!!!!!"
            d=ConfigDict.ConfigDict(self.fitresult)
            try:
                os.remove(self.outdir+"/"+self.outfile+".fit")
            except:
                pass
            d.write(self.outdir+"/"+self.outfile+".fit")
        text+= "</FONT>"
        text+= "</H2>"
        text+= "</nobr>"
        text+= "<LEFT>"
        text+= "<TABLE border=0>"
        text+= "<TR><TD><SPACER TYPE=BLOCK WIDTH=50></TD><TD>"
        text+= "<TABLE border=0 cellpadding=1 cellspacing=2>"
        text+= "  <TR><TH ALIGN=LEFT>Source : &nbsp;</TH><TD ALIGN=LEFT>%s</TD></TR>"    % (self.sourcename)
        text+= "  <TR><TH ALIGN=LEFT>Selection : &nbsp;</TH><TD ALIGN=LEFT>%s</TD></TR>" % (self.selection)
        text+= "  <TR><TH ALIGN=LEFT>Parameters : &nbsp;</TH><TD ALIGN=LEFT>"
        d=ConfigDict.ConfigDict(self.fitresult['result']['config'])
        try:
            os.remove(self.outdir+"/"+self.outfile+".txt")
        except:
            pass
        d.write(self.outdir+"/"+self.outfile+".txt")
        text+= "<a HREF=""%s"">%s</a>"% (self.outfile+".txt",self.outfile+".txt")
        text+="</TD></TR>"

        """
        text+= "  <TR><TH ALIGN=RIGHT>Source : &nbsp;</TH><TD ALIGN=LEFT>%s</TD>"%(self.sourcename)
        text+= "  <TH ALIGN=RIGHT>Selection : &nbsp;</TH><TD ALIGN=LEFT>%s</TD></TR>"%(self.selection)
        keys= [ key for key in info.keys() if key not in ['paramfile', 'peakfile'] ]
        for idx in range(0, len(keys), 2):
            text+= "  <TR><TH ALIGN=RIGHT>%s : &nbsp;</TH><TD ALIGN=LEFT>%s</TD>"%(keys[idx], info[keys[idx]])
            if idx+1<len(keys):
                text+= "  <TH ALIGN=RIGHT>&nbsp;&nbsp;%s : &nbsp;</TH><TD ALIGN=LEFT>%s</TD></TR>"%(keys[idx+1], info[keys[idx+1]])
            else:
                text+= "  <TD COLSPAN=2>&nbsp;</TD></TR>"
        """
        text+= "</TABLE>"
        text+= "</TD></TR></TABLE>"
        text+= "</LEFT>"
        return text
        
        
    def getParam(self):
        text=""
        zero = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Zero')]
        gain = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Gain')]
        noise= self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Noise')]
        fano = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Fano')]
        sum  = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Sum')]
        stdzero = self.fitresult['result']['sigmapar'][self.fitresult['result']['parameters'].index('Zero')]
        stdgain = self.fitresult['result']['sigmapar'][self.fitresult['result']['parameters'].index('Gain')]
        stdnoise= self.fitresult['result']['sigmapar'][self.fitresult['result']['parameters'].index('Noise')]
        stdfano = self.fitresult['result']['sigmapar'][self.fitresult['result']['parameters'].index('Fano')]
        stdsum  = self.fitresult['result']['sigmapar'][self.fitresult['result']['parameters'].index('Sum')]

        hypermetflag = self.fitresult['result']['config']['fit']['hypermetflag']
        if hypermetflag > 1:
            hypermetnames = ['ST AreaR', 'ST SlopeR',
                             'LT AreaR', 'LT SlopeR',
                             'STEP HeightR']
            hypermetvalues=[]
            hypermetstd   =[]
            hypermetfinalnames = []
            for name in hypermetnames:
                if name in self.fitresult['result']['parameters']:
                    hypermetvalues.append(self.fitresult['result']['fittedpar'] \
                            [self.fitresult['result']['parameters'].index(name)])
                    hypermetstd.append(self.fitresult['result']['sigmapar'] \
                            [self.fitresult['result']['parameters'].index(name)])
                    hypermetfinalnames.append(name)

        # --- html table
        text+="<H2><FONT color=#009999>"
        text+="Fit Parameters :"
        text+="</FONT></H2>"
        text+="<CENTER>"
        text+="<TABLE border=0 cellpadding=0 cellspacing=2 width=80%>"
        text+="<TR>"
        text+="    <TD><TABLE border=1 cellpadding=1 cellspacing=0 width=100%>"
        text+="        <TR align=center>"
        text+="            <TH colspan=2>FIT parameters</TH>"
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Region of Fit</I></TD>"
        text+="            <TD>&nbsp;%d - %d</TD>" % (self.fitresult['result']['config']['fit']['xmin'],self.fitresult['result']['config']['fit']['xmax'])
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Number of iterations</I></TD>"
        #text+="            <TD>&nbsp;%d</TD>" % (fitpar['fit_numiter'])
        text+="            <TD>&nbsp;%d</TD>" % (self.fitresult['result']['niter'])
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Chi square</I></TD>"
        #text+="            <TD>&nbsp;%.4f</TD>" % (fitpar['fit_chi'])
        text+="            <TD>&nbsp;%.4f</TD>" % (self.fitresult['result']['chisq'])
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Last Chi square difference</I></TD>"
        #text+="            <TD>&nbsp;%.4f %%</TD>" % (fitpar['fit_lastchi'])
        text+="            <TD>&nbsp;%.4f %%</TD>" % (self.fitresult['result']['lastdeltachi']*100)
        text+="        </TR>"
        text+="        </TABLE>"
        text+="    </TD>"
        text+="</TR>"
        text+="<TR>"
        text+="    <TD><TABLE border=1 cellpadding=1 cellspacing=0 width=100%>"
        text+="        <TR align=center>"
        text+="            <TH colspan=2>Calibration parameters</TH>"
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Zero</I></TD>"
        text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (zero, stdzero)
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Gain</I></TD>"
        text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (gain, stdgain)
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Noise</I></TD>"
        text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (noise, stdnoise)
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Fano</I></TD>"
        text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (fano, stdfano)
        text+="        </TR>"
        text+="        <TR align=left>"
        text+="            <TD><I>&nbsp;Sum</I></TD>"
        text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (sum, stdsum)
        text+="        </TR>"
        text+="        </TABLE>"
        text+="    </TD>"
        text+="</TR>"

        # --- Peak shape parameters ---
        if hypermetflag > 1:
            text+="<TR>"
            text+="    <TD><TABLE border=1 cellpadding=1 cellspacing=0 width=100%>"
            text+="        <TR align=center>"
            text+="            <TH colspan=2>Peak shape parameters</TH>"
            text+="        </TR>"
            for i in range(len(hypermetfinalnames)):
                text+="        <TR align=left>"
                text+="            <TD><I>&nbsp;%s</I></TD>" % hypermetnames[i]
                text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (hypermetvalues[i],
                                                                      hypermetstd[i])
                text+="        </TR>"
            text+="        </TABLE>"
            text+="    </TD>"
            text+="</TR>"
        

        
        # --- Continuum parameters ---
        text+="<TR>"
        text+="    <TD><TABLE border=1 cellpadding=1 cellspacing=0 width=100%>"
        text+="        <TR align=center>"
        text+="            <TH colspan=2>Continuum parameters</TH>"
        text+="        </TR>"
        # Stripping
        if self.fitresult['result']['config']['fit']['stripflag']:
             constant    = 1.0
             iterations = 20000
             if self.fitresult['result']['config']['fit'].has_key('stripconstant'):
                constant=self.fitresult['result']['config']['fit']['stripconstant']
             if self.fitresult['result']['config']['fit'].has_key('stripiterations'):
                iterations=self.fitresult['result']['config']['fit']['stripiterations']
             text+="        <TR align=left>"
             text+="            <TD><I>&nbsp;Type</I></TD>"
             text+="            <TD>&nbsp;%s</TD>" % "Strip Background"
             text+="        </TR>"
             text+="        <TR align=left>"
             text+="            <TD><I>&nbsp;%s<I></TD>" % "Strip Constant"
             text+="            <TD>&nbsp;%.5f</TD>" % constant
             text+="        </TR>"
             text+="        <TR align=left>"
             text+="            <TD><I>&nbsp;%s<I></TD>" % "Strip Iterations"
             text+="            <TD>&nbsp;%d</TD>" % iterations
             text+="        </TR>"

        # --- Background Function
        if self.fitresult['result']['config']['fit']['continuum']:
             text+="        <TR align=left>"
             text+="            <TD><I>&nbsp;Type</I></TD>"
             if self.fitresult['result']['config']['fit'].has_key('continuum_name'):
                name = self.fitresult['result']['config']['fit']['continuum_name']
                text+="            <TD>&nbsp;%s</TD>" % name
             elif self.fitresult['result']['config']['fit']['continuum'] == 1:
                text+="            <TD>&nbsp;%s</TD>" % "Constant Polymomial"
             elif self.fitresult['result']['config']['fit']['continuum'] == 2:
                text+="            <TD>&nbsp;%s</TD>" % "1st Order Polymomial"
             elif self.fitresult['result']['config']['fit']['continuum'] == 3:
                text+="            <TD>&nbsp;%s</TD>" % "2nd Order Polymomial"
             else:
                #compatibility with previous versions
                text+="            <TD>&nbsp;%s</TD>" % "1st Order Polymomial"
             text+="        </TR>"
             isum = self.fitresult['result']['parameters'].index('Sum')
             a=0
             if hypermetflag:a=5
             nglobal = len(self.fitresult['result']['parameters']) - len(self.fitresult['result']['groups'])
             for i in range(isum+1,nglobal-a):
                 text+="        <TR align=left>"
                 text+="            <TD><I>&nbsp;%s<I></TD>" % self.fitresult['result']['parameters'][i]
                 value    = self.fitresult['result']['fittedpar'][i]
                 stdvalue = self.fitresult['result']['sigmapar'] [i]
                 text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (value, stdvalue)
                 text+="        </TR>"
             if 0: 
                 text+="        <TR align=left>"
                 text+="            <TD><I>&nbsp;%s<I></TD>" % 'Constant'
                 value    = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Constant')]
                 stdvalue = self.fitresult['result']['sigmapar'] [self.fitresult['result']['parameters'].index('Constant')]
                 text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (value, stdvalue)
                 text+="        </TR>"
                 if self.fitresult['result']['config']['fit']['continuum'] > 1:
                      text+="        <TR align=left>"
                      text+="            <TD><I>&nbsp;%s<I></TD>" % 'Slope'
                      value    = self.fitresult['result']['fittedpar'][self.fitresult['result']['parameters'].index('Constant')+1]
                      stdvalue = self.fitresult['result']['sigmapar'] [self.fitresult['result']['parameters'].index('Constant')+1]
                      text+="            <TD>&nbsp;% .5E +/- % .5E</TD>" % (value, stdvalue)
                      text+="        </TR>"
             text+="</TR>"
        text+="        </TABLE>"
        text+="    </TD>"
        text+="</TR>"
        if 0:
            #not yet implemented
            text+="<TR>"
            text+="    <TD align=center>"
            text+="         <I>FIT END STATUS : </I>%s<BR>"% "STATUS"
            text+="         <B>%s</B>" % "MESSAGE"
            text+="    </TD>"
            text+="</TR>"
        text+="</TABLE>"
        text+="</CENTER>"
        return text
        
    def getFooter(self):
        now = time.time()
        text =""
        text+= "<center>"
        text+= "<table width=100%% border=0 cellspacing=0 cellpadding=0>"
        text+= "    <tr><td colspan=2 height=10><spacer type=block height=10 width=0></td></tr>"
        text+= "    <tr><td colspan=2 bgcolor=#cc0066 height=5><spacer type=block height=5 width=0></td></tr>"
        text+= "    <tr><td colspan=2 height=5><spacer type=block height=5 width=0></td></tr>"
        text+= "    <TR>"
        text+= "        <TD><FONT size=1 >created:  %s</font></TD>" % time.ctime(now)
        #text+= "        <TD ALIGN=RIGHT><FONT size=1 >last modified: %s" % time.ctime(now)
        text+= "        <TD ALIGN=RIGHT><FONT size=1 >last modified: %s by" % time.ctime(now)
        #text+= "        <A STYLE=""color: #0000cc"" HREF=""mailto:papillon@esrf.fr"">papillon@esrf.fr</A></FONT></TD>"
        if sys.platform == 'win32':
            try:
                user = os.getenv('USERNAME')
                text+= "        <A STYLE=""color: #0000cc"">%s</A></FONT></TD>" % user
            except:
                text +="</FONT></TD>"
        else:
            try:
                user = os.getenv("USER")
                text+= "        <A STYLE=""color: #0000cc"">%s</A></FONT></TD>" % user
            except:
                text +="</FONT></TD>"
        text+= "    </TR>"
        text+= "</TABLE>"
        text+= "</center>"
        text+= "</BODY>"
        text+= "</HTML>"
        return text

    def __getFitImage(self,imagefile=None):
        if imagefile is None:imagefile=self.outdir+"/"+self.outfile+".png"
        filelink = "%s" % imagefile
        text = ""
        text+= "<H2><FONT color=#009999>"
        text+= "Spectrum, Continuum and Fitted values :"
        text+= "</FONT></H2>"
        text+= "<CENTER>"
        text+= "<IMG SRC=%s ALT=""fit graph"" ALIGN=center>"%filelink
        text+= "</CENTER>"
        return text
       
    def getImage(self):
        dict=self.fitresult
 
        if MATPLOTLIB:
            fig = Figure(figsize=(6,3)) # in inches
            canvas = FigureCanvas(fig)
            ax = fig.add_axes([.1, .15, .8, .8])
            try:
                ax.grid(linestyle='--', color=0.7, linewidth=0.1)
            except:
                #above line is not supported on all matplotlib versions
                pass
            ax.set_axisbelow(True)
            ax.semilogy(dict['result']['energy'], dict['result']['ydata'], 'k', lw=1.5)
            ax.semilogy(dict['result']['energy'], dict['result']['continuum'], 'g', lw=1.5)
            ax.semilogy(dict['result']['energy'], dict['result']['yfit'], 'r', lw=1.5)
            fontproperties = FontProperties(size=8)
            if dict['result']['config']['fit']['sumflag']:
                ax.semilogy(dict['result']['energy'],
                            dict['result']['pileup'] + dict['result']['continuum'], 'y', lw=1.5)
                legend = ax.legend(('spectrum', 'continuum', 'fit', 'pileup'),0,
                                   prop = fontproperties, labelsep=0.02)
            else:
                legend = ax.legend(('spectrum', 'continuum', 'fit'),0,
                                   prop = fontproperties, labelsep=0.02)

            ax.set_xlabel('Energy')
            ax.set_ylabel('Counts')
            legend.draw_frame(False)

            outfile = self.outdir+"/"+self.outfile+".png"
            try:
                os.remove(outfile)
            except:
                pass

            canvas.print_figure(outfile)
            return self.__getFitImage(self.outfile+".png")

        if self.graph is None:
            self.widget   = qt.QWidget()
            self.widget.l = qt.QVBoxLayout(self.widget)
            self.graph  = QtBlissGraph.QtBlissGraph(self.widget)
            self.widget.l.addWidget(self.graph)
        widget = self.widget
        graph  = self.graph
            
        graph.xlabel('Energy')
        graph.ylabel('Counts')
        graph.setCanvasBackground(qt.Qt.white)
        x = dict['result']['energy']
        graph.newcurve('spectrum', x=x,y=dict['result']['ydata'],logfilter=1)
        graph.newcurve('continuum',x=x,y=dict['result']['continuum'],logfilter=1)
        graph.newcurve('fit',x=x,y=dict['result']['yfit'],logfilter=1)
        if dict['result']['config']['fit']['escapeflag']:
            #I DO NOT HAVE THE CONTRIBUTION
            pass
            #self.graph.newcurve('escape',x=x,y=dict['result']['escape'],logfilter=1)
        if dict['result']['config']['fit']['sumflag']:
            graph.newcurve('pileup',
                                x=x,
                                y=dict['result']['pileup']+dict['result']['continuum'],
                                logfilter=1)                            
        graph.ToggleLogY()
        ymin=min(min(dict['result']['ydata']),min(dict['result']['yfit']))
        ymax=max(max(dict['result']['ydata']),max(dict['result']['yfit']))
        graph.sety1axislimits(ymin,ymax)
        graph.sety2axislimits(ymin,ymax)
        graph.show()
        if True or qt.qVersion() < '3.0.0':
            widget.resize(450,300)
            #widget.show()
        
        qt.qApp.processEvents()
        outfile = self.outdir+"/"+self.outfile+".png"
        pixmap = qt.QPixmap.grabWidget(widget)
        try:
            os.remove(outfile)
        except:
            pass
        if pixmap.save(outfile,'PNG'):
            qt.qApp.processEvents()
            graph.close()
            del graph
            widget.close()
            del widget
            return self.__getFitImage(self.outfile+".png")
        else:
            print "cannot generate image"
            return ""

    def getConcentrations(self):
        text = ""
        if self.concentrations is None:return text
        text+="\n"
        text+= "<H2><a NAME=""%s""></a><FONT color=#009999>" % 'Concentrations'
        text+= "%s:" % 'Concentrations'
        text+= "</FONT></H2>"
        text+="<br>"
        result =self.concentrations
        #the header
        
        #the table
        labels = ['Element','Group','Fit Area','Sigma Area', 'Mass fraction']
        if result.has_key('layerlist'):
            if type(result['layerlist']) != type([]):
                result['layerlist'] = [result['layerlist']]
            for label in result['layerlist']:
                labels += [label]
        lemmon=string.upper("#%x%x%x" % (255,250,205))
        white ='#FFFFFF' 
        hcolor = string.upper("#%x%x%x" % (230,240,249))       
        text+="<CENTER>"
        text+=("<nobr>")
        text+=( "<table WIDTH=80%%")
        text+=( "<tr>")
        for l in labels:
            text+=('<td align="left" bgcolor="%s"><b>' % hcolor)
            text+=l
            text+=("</b></td>")
        text+=("</tr>")
        line = 0
        for group in result['groups']:
            text+=("<tr>")
            element,group0 = string.split(group)
            fitarea    = "%.6e" % result['fitarea'][group]
            sigmaarea  = "%.2e" % result['sigmaarea'][group]
            area       = "%.6e" % result['area'][group]
            fraction   = "%.4g" % result['mass fraction'][group]
            if 'Expected Area' in labels:
                fields = [element,group0,fitarea,sigmaarea,area,fraction]
            else:
                fields = [element,group0,fitarea,sigmaarea,fraction]
            if result.has_key('layerlist'):
                for layer in result['layerlist']:
                    #fitarea    = qt.QString("%.6e" % (result[layer]['fitarea'][group]))
                    #area       = qt.QString("%.6e" % (result[layer]['area'][group]))
                    if result[layer]['mass fraction'][group] < 0.0:
                        fraction   = "Unknown"
                    else:
                        fraction   = "%.4g" % result[layer]['mass fraction'][group]
                    fields += [fraction]
            if line % 2:
                color = lemmon
            else:
                color = white
            i = 0 
            for field in fields:
                if (i<2):
                    #text += '<td align="left"  bgcolor="%s"><b>%s</b></td>' % (color, field)
                    text += '<td align="left"  bgcolor="%s">%s</td>' % (color, field)
                else:
                    #text += '<td align="right" bgcolor="%s"><b>%s</b></td>' % (color, field)
                    text += '<td align="right" bgcolor="%s">%s</td>' % (color, field)
                i+=1
            text += '</tr>'
            line +=1           
        text+=("</table>")
        text+=("</nobr>")
        text+="</CENTER>"
        return text        

    def getConcentrationsASCII(self):
        text = ""
        if self.concentrations is None:return text
        result =self.concentrations
        #the header
        
        #the table
        labels = ['Element','Group','Fit_Area','Sigma_Area', 'Mass_fraction']
        if result.has_key('layerlist'):
            if type(result['layerlist']) != type([]):
                result['layerlist'] = [result['layerlist']]
            for label in result['layerlist']:
                labels += [label.replace(' ','')]
        for l in labels:
            text+="%s  " % l
        text+=("\n")
        line = 0
        for group in result['groups']:
            element,group0 = string.split(group)
            fitarea    = "%.6e" % result['fitarea'][group]
            sigmaarea  = "%.2e" % result['sigmaarea'][group]
            area       = "%.6e" % result['area'][group]
            fraction   = "%.4g" % result['mass fraction'][group]
            if 'Expected Area' in labels:
                fields = [element,group0,fitarea,sigmaarea,area,fraction]
            else:
                fields = [element,group0,fitarea,sigmaarea,fraction]
            if result.has_key('layerlist'):
                for layer in result['layerlist']:
                    #fitarea    = qt.QString("%.6e" % (result[layer]['fitarea'][group]))
                    #area       = qt.QString("%.6e" % (result[layer]['area'][group]))
                    if result[layer]['mass fraction'][group] < 0.0:
                        fraction   = "Unknown"
                    else:
                        fraction   = "%.4g" % result[layer]['mass fraction'][group]
                    fields += [fraction]
            i = 0 
            for field in fields:
                text += '%s  ' % (field)
                i+=1
            text += '\n'
            line +=1
        return text        

    def getResult(self):
        text = ""
        if self.tableFlag == 0:return text
        text+="\n"
        text+= "<H2><a NAME=""%s""></a><FONT color=#009999>" % 'Fit_Peak_Results'
        text+= "%s:" % 'Fit Peak Results'
        text+= "</FONT></H2>"
        text+="<br>"
        result = self.fitresult['result']
        labels=['Element','Group','Fit Area','Sigma','Energy','Ratio','FWHM','Chi square']   
        lemmon=string.upper("#%x%x%x" % (255,250,205))
        hcolor = string.upper("#%x%x%x" % (230,240,249))       
        text+="<CENTER>"
        text+=("<nobr>")
        text+=( "<table WIDTH=80%%")
        text+=( "<tr>")
        for l in labels:
            text+=('<td align="left" bgcolor="%s"><b>' % hcolor)
            text+=l
            text+=("</b></td>")
        text+=("</tr>")
        for group in result['groups']:
            text+=("<tr>")
            ele,group0 = string.split(group)
            text += '<td align="left"><b>%s</b></td>' % ele
            text += '<td align="left"><b>%s</b></td>' % group0
            fitarea    = "%.6e" % result[group]['fitarea']
            sigmaarea  = "%.2e" % result[group]['sigmaarea']
            text += '<td align="right"><b>%s</b></td>' % fitarea
            text += '<td align="right"><b>%s</b></td>' % sigmaarea
            text += '</tr>'
            if type(result[group]['peaks']) != type([]):
                iterator = [result[group]['peaks']]
            else:
                iterator = 1 * result[group]['peaks']
            for peak in iterator:
                text += '<tr><td></td>'
                name  = peak
                energy = ("%.3f" % (result[group][peak]['energy']))
                ratio  = ("%.5f" % (result[group][peak]['ratio']))
                area   = ("%.6e" % (result[group][peak]['fitarea']))
                sigma  = ("%.2e" % (result[group][peak]['sigmaarea']))
                fwhm   = ("%.3f" % (result[group][peak]['fwhm']))
                chisq  = ("%.2f" % (result[group][peak]['chisq']))
                fields = [name,area,sigma,energy,ratio,fwhm,chisq]
                for field in fields:
                    if field == name:
                        text+=('<td align="left"  bgcolor="%s">%s</td>' % (lemmon,field))
                    else:
                        text+=('<td align="right" bgcolor="%s">%s</td>' % (lemmon,field))
                text+="</tr>"
            if type(result[group]['escapepeaks']) != type([]):
                iterator = [result[group]['escapepeaks']]
            else:
                iterator = 1 * result[group]['escapepeaks']
            for peak0 in iterator:
                name  = peak0+"esc"
                peak  = peak0+"esc"
                if result[group][name]['ratio'] > 0.0:
                    text += '<tr><td></td>'
                    energy = ("%.3f" % (result[group][peak]['energy']))
                    ratio  = ("%.5f" % (result[group][peak]['ratio']))
                    area   = ("%.6e" % (result[group][peak]['fitarea']))
                    sigma  = ("%.2e" % (result[group][peak]['sigmaarea']))
                    fwhm   = ("%.3f" % (result[group][peak]['fwhm']))
                    chisq  = ("%.2f" % (result[group][peak]['chisq']))
                    fields = [name,area,sigma,energy,ratio,fwhm,chisq]
                    for field in fields:
                        if field == name:
                            text+=('<td align="left"  bgcolor="%s">%s</td>' % (lemmon,field))
                        else:
                            text+=('<td align="right" bgcolor="%s">%s</td>' % (lemmon,field))
                    text+="</tr>"
        text+=("</table>")
        text+=("</nobr>")
        text+="</CENTER>"
        return text


def generateoutput(fitfile,outfile=None):
    report = QtMcaAdvancedFitReport(fitfile, outfile)
    report.writeReport()

if __name__ == "__main__":
    if len(sys.argv) <2 :
        print "Usage: %s Input_Fit_Result_File [optional_output_file]" % sys.argv[0]
        sys.exit(1)
    app = qt.QApplication(sys.argv)
    fitfile=sys.argv[1]
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    else:
        outfile = None
    generateoutput(fitfile,outfile)
    app.quit() 
    
