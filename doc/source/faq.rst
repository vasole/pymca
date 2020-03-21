Frequently Asked Questions
==========================

- `Should I write PyMCA or PyMca?`_
- `Why did you take the Gioconda as logo?`_
- `I do not use the ESRF data format nor the SPEC file format. Do I have to convert my data?`_
- `I use an X-ray tube, how can I make quantitative analysis?`_
- `The description of the scattering peaks is very poor, why?`_
- `I am on windows, what program version should I use?`_
- `Does PyMca work on Windows 7 and Windows 10?`_
- `I have a Mac, the program seems to hang or to do nothing, how can I report what's happening?`_
- `What have you used to build the binaries?`_
- `I want to build the program from its source code. What do I need?`_

Should I write PyMCA or PyMca?
------------------------------

It's up to you. The program has been published as PyMCA because of the scientific use of MCA for multichannel analyzer but PyMca is more pythonic and it is what you had to type to get the program running. Due to some problems I encountered with the publisher, I have some preference for PyMca because it makes clear that it is the name of the program and it does not intend to be an acronym.

Why did you take the Gioconda as logo?
--------------------------------------

Believe it or not, Mona Lisa has more to do with the PyMca code than with The Da Vinci Code. In particular, the support of multilayered samples and of X-ray tubes was greatly influenced by the use of PyMca by the Centre de Recherche et Restauration des Musees de France (C2RMF) to analyze X-ray spectra obtained from that master piece. The results of that work were published in July 2010.

I do not use the ESRF data format nor the SPEC file format. Do I have to convert my data?
-----------------------------------------------------------------------------------------

Probably not. Most common formats are wrapped by PyMca as SPEC file format. That includes multicolumn ASCII, Canberra's .TK, AmpTek, and QXAS. If your format is not supported but you know how to read it, it should not be a big problem to implement it.

Starting with version 4.4.0, PyMca supports the HDF5 format. Due to its versatility, this format will progresively become the preferred input and output format of PyMca.

I use an X-ray tube, how can I make quantitative analysis?
----------------------------------------------------------

Well, you will have to characterize your tube or, better said, find a description of it in terms of discrete energies that allows you to reproduce the concentrations of a set of calibrated standards. These standards have to cover your energy range of interest and you should give first priority to K shell standards and secondly to L shell standards.

The supplied X-ray tube profile calculation tool is just for guidance and, unless you are going to measure samples that are very similar to your standards, I really doubt that you can use the generated profile without some "hand work". In any case, please consider ALL sources of attenuation between the beam and the sample and between the sample and the detector. If you aim to work at very low energies, please consider the atmosphere between the detector window and the detector itself. Some detectors are not under vacuum but under some inert gas atmosphere.

In its simplest form the procedure would consist on measuring a thin film standard and entering as matrix composition the known composition of the standard. When asking the program to calculate the concentrations using a matrix element as reference, it should give the exact concentration at least for the reference element. Then, at the concentrations tab,  switch to the fundamental parameter method. Adjust the time and solid angle parameters to match those of the measurement. At that point, start to play with the flux parameter till you reproduce the same result as the one obtained with the internal reference. Once you have found a set of fundamental parameters that reproduce all your standards within the desired accuracy, you will be ready. The procedure can be/is tedious, but it is really worth the effort.

The description of the scattering peaks is very poor, why?
----------------------------------------------------------

Because they are fitted as simple gaussians and that is not correct. The reason? I considered that you could always leave them out of the fitting region and therefore the main interest of having them is to allow to deal with their escape peaks when falling  into the fitting region. I have to admit that, while this is mostly the case when using synchrotron radiation as excitation source, the Compton peaks can be a problem when using X-ray tubes.

I am on windows, what program version should I use?
---------------------------------------------------

Always the most recent. If you encounter any problem using a previous version, please verify the problem is still present when using the latest release.

I have a Mac, the program seems to hang or to do nothing, how can I report what's happening?
--------------------------------------------------------------------------------------------

In most of the platforms I leave a console open in order to catch there unhandled error messages that can help to debug problems. To have such information on the Mac you may need to run the program from a terminal. If you have your application on your desktop, you should open a terminal window and type:

./Desktop/PyMca4.3.0.app/Contents/MacOS/PyMca4.3.0

to start the application from the console and see any possible error output there. Of course, you will have to replace 4.3.0 by the number of your PyMca version.

What have you used to build the binaries?
-----------------------------------------

I have used cx_freeze on linux and windows. For the Mac I have used py2app. In order to make installable packages I have used the Nullsoft installer on windows and Platypus on the Mac.

I want to build the program from its source code. What do I need?
-----------------------------------------------------------------

Please refer to the paragraph :ref:`Installing from source` in the installation instructions.
