XRF Analysis
============

.. |img1| image:: ./img/tertiary_01.png
   :width: 300px
   :align: middle
   :alt: Secondary Excitation Process

.. |img2| image:: ./img/tertiary_02.png
   :width: 400px
   :align: middle
   :alt: Fitted Steel Data

.. |img3| image:: ./img/tertiary_03.png
   :width: 400px
   :align: middle
   :alt: Analytical corrections

.. |img4| image:: ./img/tertiary_04.png
   :width: 400px
   :align: middle
   :alt: Monte Carlo corrections

.. contents::
   :local:

Introduction
------------

Many users coming to synchrotrons to perform X-ray fluorescence experiments focus on imaging the distribution of elements in the sample and they show little interest in learning how to perform quantitative X-ray fluorescence (XRF) analysis.

In fact, the additional step to take to pass from pure qualitative to quantitative analysis is a very small one. One has to take into account that even for pure imaging experiments one needs to calibrate the spectra, to identify the different elements in the sample and, most of the times, perform some fitting in order to resolve overlapping peaks of different elements. At this point the might not be aware that the relative peak areas they are extracting may be wrong because they did not take into account the modification of the theoretical peak ratios by the conditions of the experiment.

Simply introducing the experimental conditions and a guestimate of the sample composition is often enough to properly extract the signal from the different elements. At this point, quantitative analysis just comes as a by-product.

Many X-ray fluorescence experiments at synchrotrons focus on imaging the distribution of elements in the sample. It is not uncommon that users just calibrate their spectra and proceed to identify elements and either assign regions of interest or perform  qualitative fits to obtain beautiful maps. However, they might not be aware that the relative peak areas they are extracting may be wrong because they did not take into account the modification of the theoretical peak ratios by the conditions of the experiment.


Exercise
--------

The objective of this exercise is to get familiar with x-ray fluorescence analysis. For this, we are going to work with a spectrum frmom a thin film standard. However, we are going to make some simplifications.

Step 1: Loading the data
........................

The data required for this exercise are supplied with PyMca and can be loaded into the program via the main window File menu following the sequence File->Open->Load Training Data->XRF Spectrum.

The format associated to that spectrum is the simplest that PyMca can read. It is just a single column of numbers corresponding to the counts in the different channels. Under that situation, PyMca does not know if those data belong to an XRF experiment or to something else and offers two different visualization modes. One generic and one specific to XRF. 

Your first exercise is to achieve the situation shown in the figure below where the data are present in the MCA tab of the main window in a semilogarithmic plot.

|img1|

Step 2: Calibrating the data
............................

If it is your first time with PyMca, you should take a look at the `Calibration tutorial <http://www.esrf.fr/computing/bliss/downloads/pymca/calibrationtutorial.htm>`_

The excitation energy was about 17.5 keV. Very often this is enough information for an initial calibration. However, this detector presented a very important offset and you will need an addition calibration point. Just imagine you have previously measured a cobalt sample and that you know that the peak around channel 1474 corresponds to the main emission line of Co.

You may reach the situation illustrated below where the calibration window is shown. You have to press the OK button to validate the calibration.

|img 2|

At this point you should be back to the main window without any change respect to the previous situation. Prior to go any further, you should instruct PyMca about what calibration you intend to use. Unless you have changed the name of the calibration, choosing Internal in the calibration combo box should apply the just calculated one to the spectrum leading to the situation below.

|img 3|

Under the calibration combo box, following *Active curve uses* , you will see the calibration actually applied. It should be close t0 A=-0.5, B=-0.005 and C=0. (Hint: Make sure you have selected a first order calibration when calculating the spectrum). It it isvery different your calibration is wrong and you will experience a lot of difficulties later on.


Step 3: Select your fit region
..............................

At this point we have a calibrated spectrum. The rest of the exercise will use the McaAdvancedFit window.

Prior to reach that window, we should select the region of the sample we'd like to analyze. For that, whe have to zoom in that region by pressing and dragging the mouse. PyMca implements a zoom stack, you can go back by pressing the mouse right button or by pressing the the reset zoom icon.

At the very least, you should always leave the cut at the low energy side corresponding to the low-level discriminator of your acquisition system out of the fitting region. Something around 1.0 keV should be OK in this case.

PyMca (still!) implements a very poor description of the scattering peaks. Unless you absolutely need it, you will obtain better results by limiting the high energy side of the region to the rail of the scattered peaks. Something like 16.3 keV should be a good upper limit.

|img 4|

At this point we are ready to access the McaAdvancedFit window by pressing the fit icon and selecting the *Advanced* option.

Step 4: Using the Peak Identifier
.................................

The first thing you will get is a message telling you that no peaks have been defined. PyMca has very good peak search routines and it could do a very good guess about the elements present. However, the author(s) consider that the responsibility should fall on the person carrying the analysis.

In order to allow PyMca to give you some hints about what elements can be associatedd to a peak, you need to toggle the energy axis on. Your next target should be to obtain the image below.

|img 5|

If you now click on top of a peak, PyMca will show you the peaks that can be associated to that energy. If you click at around 6.9 keV. PyMca should show you the peak identifier.

|img 6|


The experimental conditions are excitation energy around 17.5 keV, Si detector 450 micron thickness and Be window of 8 micron thickness. For the sake of simplicity assume the sample is 100 micron water and contains 500 ppm of Co. Incident beam angle is 0.1 degrees and fluorescence beam angle is 90 degrees. There is an air path between sample and detector window of 2 mm.



Exercise: Analyze a thin film standard. Spectrum nov07e18.mca

Experimental conditions: Excitation energy around 17.5 keV, Si detector 450 micron thickness and Be window of 8 micron. For the sake of simplicity assume the sample is 100 micron water and contains 500 ppm of Co. Incident beam angle is 0.1 degrees and fluorescence beam angle is 90 degrees. There is an air path between sample and detector window of 2 mm.

Steps: 	1 – Calibrate the data. This detector presented a huge energy offset.
 Main Co peak is found close to channel 1474.
	2 – Enter as much information as you know in the fit configuration.
	3 – Iteratively, identify peaks and perform fits.
	4 – When you are satisfied, perform a quantification using an internal standard.
	5 – Make the necessary modifications to work in fundamental parameters method.
	6 – Using the matrix spectrum, guess what would be the smallest amount of a given element (for instance Sc) that you would be able to quantify under the conditions the measurement took place.
	7 – If time allows it: Observe the change on the calculated concentrations when changing the attenuation conditions:
     - play with an air path between 1.0 mm and 100 mm (what happens at low energies?)
     - play with a detector thickness between 10 micron and 1 mm (what happens with the concentrations at high energies?) 

