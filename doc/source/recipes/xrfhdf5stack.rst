HDF5 XRF Stack
==============

There is a recurrent cuestion concerning how one should write the spectra associated to a raster
experiment to be compatible with PyMca.

The solution is not unique, because PyMca can deal with (too) many data formats. For instance, if
you have a map of 20000 spectra corresponding to a map of 100 rows per 200 columns, 20000 single-column
ASCII files containing the measured counts would do the job. You would be "compatible" with PyMca but
you would be missing relevant information known at acquisition time like live_time and calibration 
parameters.

The most versatile file format supported by PyMca is without doubt HDF5. You can find information about it
at the `HDF Group web site <https://portal.hdfgroup.org/display/HDF5/HDF5>`_ 

Let's assume data is a 3-dimensional array or 20000 spectra corresponding to a raster scan of 100 rows per
200 columns. If each spectrum has 2048 channels, the shape of that array will be (following C-convention)
(100, 200, 2048). The simplest HDF5 file compatible with PyMca would contain a single 3-dimensional dataset
and it could be written using the code snipset shown below.

.. code-block:: python

    import h5py
    h5 = h5py.File("myfile.h5", "w")
    h5["data"] = data
    h5.flush()
    h5.close()

Obviously, besides a faster readout of the data by PyMca, one would not gain any information compared to
the use of single-column ASCII files.

PyMca will automatically look for information associated to a dataset provided that information is
stored within the same container group in the file. 

If live_time is a one dimensional dataset with 20000 values corresponding to the actual measuring time 
associated to each spectrum, the simplest way to allow PyMca to use that information is to put it at 
the same level within the same group.

If the channels associated to the data are different from 0,1,2,3, ..., 2046, 2047, they can be specified by
a one dimensional dataset named channels.

The calibration can be specified as a dataset containing three values (corresponding to a, b and c in the
expression energy = a + b * ch + c * ch^2 and named calibration.

.. code-block:: python

    import h5py
    h5 = h5py.File("myfile.h5", "w")
    h5["/mca_0/data"] = data
    h5["/mca_0/channels"] = channels
    h5["/mca_0/calibration"] = calibration
    h5["/mca_0/live_time"] = live_time
    h5.flush()
    h5.close()

Additional conventions can be applied to improve the user experience when using the PyMca graphical user
interface.

The code below writes an HDF5 following NeXus conventions. Those conventions are attribute based, therefore
the actual names of the different groups are free. You can :download:`download a script <./recipescode/GenerateHDF5Stack.py>` generating a file using these 
conventions.

.. code-block:: python

    h5File = "myfile.h5"
    if os.path.exists(h5File):
        os.remove(h5File)
    h5 = h5py.File(h5File, "w")
    h5["/entry/instrument/detector/calibration"] = calibration
    h5["/entry/instrument/detector/channels"] = channels
    h5["/entry/instrument/detector/data"] = data
    h5["/entry/instrument/detector/live_time"] = live_time

    # add nexus conventions (not needed)
    h5["/entry/title"] = u"Dummy generated map"
    h5["/entry"].attrs["NX_class"] = u"NXentry"
    h5["/entry/instrument"].attrs["NX_class"] = u"NXinstrument"
    h5["/entry/instrument/detector/"].attrs["NX_class"] = u"NXdetector"
    h5["/entry/instrument/detector/data"].attrs["interpretation"] = u"spectrum"
    # implement a default plot named measurement (not needed)
    h5["/entry/measurement/data"] = h5py.SoftLink("/entry/instrument/detector/data")
    h5["/entry/measurement"].attrs["NX_class"] = u"NXdata"
    h5["/entry/measurement"].attrs["signal"] = u"data"
    h5["/entry"].attrs["default"] = u"measurement"

    h5.flush()
    h5.close()
