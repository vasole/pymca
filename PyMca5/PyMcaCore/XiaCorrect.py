#/*##########################################################################
# Copyright (C) 2004-2014 E. Papillon, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "E. Papillon - ESRF Software group"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from . import XiaEdf
import sys
import os
import time

__version__="$Revision: 1.11 $"

def defaultErrorCB(message):
    print(message)

def defaultLogCB(message, verbose_level=None, verbose_ask=None):
    if verbose_level is None:
        print(message)
    elif verbose_level <= verbose_ask:
        print(message)

def defaultDoneCB(nbdone, total):
    pass

def checkCB(log_cb=None, done_cb=None, error_cb=None):
    if log_cb is None:
        log_cb= defaultLogCB
    if done_cb is None:
        done_cb= defaultDoneCB
    if error_cb is None:
        error_cb= defaultErrorCB

    return (log_cb, done_cb, error_cb)


def parseFiles(filelist, verbose=0, keep_sum=0, log_cb=None, done_cb=None, error_cb=None):
    (log_cb, done_cb, error_cb)= checkCB(log_cb, done_cb, error_cb)

    log_cb("Checking xia files ...")
    xiafiles= []

    for file in filelist:
        xf= XiaEdf.XiaFilename(file)
        if xf.isValid():
            log_cb(" - Parsing %s (OK - %s)"%(file, xf.getType()), 1, verbose)
            if not keep_sum:
                if not xf.isSum():
                    xiafiles.append(xf)
            else:
                xiafiles.append(xf)
        else:
            log_cb(" - Parsing %s (Not Xia)"%file, 1, verbose)

    if len(xiafiles):
        log_cb("Sorting xia files ...")
        xiafiles.sort()

        groupfiles= []
        group= None

        for xf in xiafiles:
            if group is None:
                group= [ xf ]
            else:
                if xf.isGroupedWith(group[0]):
                    group.append(xf)
                else:
                    groupfiles.append(group)
                    group= [ xf ]
        if group is not None:
            groupfiles.append(group)

        grouperrors= []
        for group in groupfiles:
            if group[0].isScan():
                if not group[-1].isStat():
                    stat= group[0].findStatFile()
                    if stat is not None:
                        log_cb(" - Find stat file for group <%s>"%stat.get(), 1, verbose)
                        group.append(stat)
                    else:
                        error_cb("XiaCorrect ERROR: no stat file in current group <%s>"%group[0].get())
                        grouperrors.append(group)

        for group in grouperrors:
            groupfiles.remove(group)

        if not len(groupfiles):
            error_cb("XiaCorrect ERROR: No valid XIA group files")
            return None

        return groupfiles

    else:
        error_cb("XiaCorrect ERROR: No XIA files found.")
        return None


def correctFiles(xiafiles, deadtime=1, livetime=0, sums=None, avgflag=0, outdir=None, outname="corr", force=0, \
		    verbose=0, log_cb=None, done_cb=None, error_cb=None):
    (log_cb, done_cb, error_cb)= checkCB(log_cb, done_cb, error_cb)

    processed= 0
    saved= 0
    total= 0
    errors= 0
    tps= time.time()

    done_cb(0, total)
    total= len(xiafiles)

    log_cb("Correcting xia files ...")

    for group in xiafiles:
        if not group[0].isScan():
            file= group[0]
            name= file.get()
            log_cb("Working on %s"%name, 1, verbose)

            try:
                xia= XiaEdf.XiaEdfCountFile(name)
                file.setDirectory(outdir)
                file.appendPrefix(outname)
                name= file.get()

                if sums is not None:
                    err= xia.sum(sums, deadtime, livetime, avgflag)
                    file.setType("sum", -1)
                else:
                    err= xia.correct(deadtime, livetime)
                if len(err):
                    error_cb(" - WARNING: in %s"%name)
                    for msg in err:
                        error_cb("     * " + msg)

                log_cb(" - Saving %s"%name)
                xia.save(name, force)
                saved += 1

            except XiaEdf.XiaEdfError:
                errors += 1
                log_cb(sys.exc_info()[1])

        else:
            groupfiles= [ file.get() for file in group ]
            name= groupfiles[-1]
            log_cb("Reading %s"%name, 1, verbose)

            try:
                xia= XiaEdf.XiaEdfScanFile(name, groupfiles[:-1])
            except XiaEdf.XiaEdfError:
                xia= None
                errors += 1
                error_cb(sys.exc_info()[1])

            if xia is not None:
                for file in group:
                    file.setDirectory(outdir)
                    file.appendPrefix(outname)

                if sums is None:
                    for file in group[:-1]:
                        det= file.getDetector()

                        if det is not None:
                            log_cb("Working on detector #%02d"%det, 1, verbose)
                            try:
                                err= xia.correct(det, deadtime, livetime)
                                name= file.get()

                                if len(err):
                                    error_cb(" - WARNING: in %s"%name)
                                    for msg in err:
                                        error_cb("     * " + msg)

                                log_cb(" - Saving %s"%name)
                                xia.save(name, force)

                                saved += 1

                            except XiaEdf.XiaEdfError:
                                errors += 1
                                error_cb(sys.exc_info()[1])
                else:
                    log_cb("Working on group %s"%name, 1, verbose)
                    file= group[-1]
                    for isum in range(len(sums)):
                        try:
                            err= xia.sum(sums[isum], deadtime, livetime, avgflag)

                            file.setType("sum", isum+1)
                            name= file.get()

                            if len(err):
                                error_cb(" - WARNING: in %s"%name)
                                for msg in err:
                                    error_cb("     * " + msg)

                            log_cb(" - Saving %s"%name)
                            xia.save(name, force)

                            saved += 1
                        except XiaEdf.XiaEdfError:
                            errors += 1
                            error_cb(sys.exc_info()[1])

        processed += 1
        done_cb(processed, total)

    done_cb(total, total)
    log_cb("\n* %d groups processed and %d files saved in %.2f sec"%(processed, saved, time.time()-tps))
    if not errors:
        log_cb("* No errors found")
    else:
        log_cb("* %d errors found"%errors)
    log_cb("\n")


def parseArguments():
    import getopt, os.path

    prog= os.path.basename(sys.argv[0])

    long = ["help", "input=", "output=", "force", "verbose", "deadtime", "livetime", "sum=", "avg", "name=", "parsing"]
    short= ["h",    "i:",     "o:",      "f",     "v",       "d",        "l",        "s:",   "a",   "n:",    "p"]

    try:
        opts, args= getopt.getopt(sys.argv[1:], " ".join(short), long)
    except getopt.error:
        print("XiaCorrect ERROR: Cannot parse command line arguments")
        print("\t%s" % sys.exc_info()[1])
        sys.exit(0)

    parsing= 0
    options= {"input": [], "files": [], "output": None, "force": 0, "name": "corr",
		"verbose": 0, "deadtime": 0, "livetime": 0, "sums": None, "avgflag": 0, "parsing": 0}

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printHelp()
            sys.exit(0)
        if opt in ("-i", "--input"):
            options["input"].append(os.path.normpath(arg))
        if opt in ("-o", "--output"):
            options["output"]= os.path.normpath(arg)
        if opt in ("-f", "--force"):
            options["force"]= 1
        if opt in ("-v", "--verbose"):
            options["verbose"]= 1
        if opt in ("-d", "--deadtime"):
            options["deadtime"]= 1
        if opt in ("-l", "--livetime"):
            options["livetime"]= 1
        if opt in ("-n", "--name"):
            options["name"]= str(arg)
        if opt in ("-s", "--sum"):
            if options["sums"] is None:
                options["sums"]= []
            try:
                ssum= [ int(det) for det in arg.split(",") ]
                if ssum[0]==-1:
                    ssum= []
                options["sums"].append(ssum)
            except:
                print("XiaCorrect ERROR: Cannot parse sum detectors")
                print("\t%s"%arg)
                sys.exit(0)
        if opt in ("-a", "--avg"):
            options["avgflag"]= 1
        if opt in ("-p", "--parsing"):
            options["parsing"]= 1


    for iinput in options["input"]:
        if not os.path.isdir(iinput):
            print("XiaCorrect WARNING: Input directory <%s> is not valid"%\
                  iinput)

        files= [ os.path.join(iinput, file) for file in os.listdir(iinput) ]
        if not len(files):
            print("XiaCorrect WARNING: Input directory <%s> is empty"%\
                  (iinput, prog))
        else:
            options["files"]+= files

    if len(args):
        options["files"]+= args

    if not len(options["files"]):
        print("XiaCorrect ERROR: No input datafiles")
        sys.exit(0)

    if not options["parsing"]:
        if not options["deadtime"] and not options["livetime"] and options["sums"] is None:
            print("XiaCorrect ERROR: Must have at least deadtime, livetime or sum options")
            sys.exit(0)

        if options["output"] is not None:
            if not os.path.isdir(options["output"]):
                print("XiaCorrect ERROR: output directory is not valid")
                sys.exit(0)

    return options

def printHelp():
    prog= os.path.basename(sys.argv[0])
    msg= """

%s [-h] [-v] [-f] [-d] [-l] [-a] [-s <detlist>] [-i <directory>] [-o <directory>] [<files ...>]

Options:
    [-h]/[--help]
            Print help message
    [-v]/[--verbose]
            Switch ON verbose mode
    [-f]/[--force]
            Force writing output files if they already exists
    [-d]/[--deadtime]
            Perform deadtime correction
    [-l]/[--livetime]
            Perform livetime normalization
    [-s]/[--sum] <detector_list_comma_separated>
            Sum given detectors. if detector list is set to (-1),
            all detectors are used:
                %s -s 2,4,8  --> will sum detectors 2,4 and 8
                %s -s -1     --> will sum ALL detectors
	    Several sums can be added:
		-s 2,4,6,7 -s 8,9,10,11
    [-a]/[--avg]
	    Sum(s) are averaged.
	    Need <-s> to specify list of detectors:
		-s 2,3,4 -a  --> will average detectors 2,3 and 4
    [-i]/[--input] <directory>
            Specify input directory: all files in this directory
            which appears to be xia edf files are processed.
            Several [-i] options can be added:
                %s -d -i /tmp -i /data/opidXX
    [-o]/[--output]
            Specify output directories. If not specified, output
            files are saved in the same place as input file.
    [-n]/[--name]
	    String to be appended to prefix for output filename.
	    Default is \"corr\".
    [<files ...>]
            Specify one or several input files. Wildcards can be used:
                %s -l file1.edf file2.edf /tmp/test*.edf

Minimum options to work:
    [-l] , [-d] or [-s]
    [-i input] or <files ...>

"""%(prog, prog, prog, prog, prog)
    print(msg)


def mainCommandLine():
    options= parseArguments()
    files= parseFiles(options["files"], options["verbose"])
    if files is not None:
        if options["parsing"]:
            for group in files:
                print("FileGroup:")
                for file in group:
                    print(" - ", file.get())
        else:
            correctFiles(files, options["deadtime"], options["livetime"], options["sums"], options["avgflag"], \
                 options["output"], options["name"], options["force"], options["verbose"])

def mainGUI(app=None):
    from PyMca5.PyMcaGui import PyMcaQt as qt
    from PyMca5.PyMcaGui.pymca import XiaCorrectWizard

    if app is None:
        app= qt.QApplication(sys.argv)

    wid= XiaCorrectWizard.XiaCorrectWizard()
    ret= wid.exec()

    if ret==qt.QDialog.Accepted:
        options= wid.get()
        files= parseFiles(options["files"], options["verbose"])
        if files is not None:
            correctFiles(files, options["deadtime"], options["livetime"], options["sums"], options["avgflag"], \
                     options["output"], options["name"], options["force"], options["verbose"])




if __name__=="__main__":
    import sys

    if len(sys.argv)==1:
        mainGUI()
    else:
        mainCommandLine()

