Accessing HDF5 Data
===================

Version 4.4.0 of PyMca introduced `HDF5 <https://portal.hdfgroup.org/display/HDF5/HDF5>`_ file format support using Andrew Collette's `h5py <https://www.h5py.org/>`_ as Python wrapping library.

For those interested, a simple analogy of an HDF5 file is that of a hard disk.  A hard disk can contain files that can be into folders that in turn may contain other folders. An HDF5 file contains datasets (your data) that can be arranged into groups that in turn may contain other groups. The analogy goes till the point that you can create links between datasets or groups and that to access a dataset or a group you have to provide the path to it.

Obviously, from a graphical user interface point of view, the logical access to an HDF5 should be provided by something similar to a file browser. 

The HDF5 file browser used in PyMca is based on a contribution by Darren Dale.

Generic HDF5 Support
--------------------

The data in an HDF5 file provide information about their size and type but they do not provide information about what they represent. Therefore, the approach followed by PyMca to properly visualize the data is cumbersome (at least when used for first time) but simple. 

The approach is based on creating a USER selection table with the datasets of interest in order to allow the user to choose what to visualize (aka. *signal*) against what (aka. *axes*).

This can be achieved by double clicking the relevant datasets or via a right-button mouse click. 
The nice feature is that the table provides a context menu (right-buttonmouse click) allowing the user to save or load selection tables therefore reducing the need to repetitively browse the file. In addition, the selection table is saved among the PyMca settings (File Menu -> Save ->PyMca Configuration or File Menu -> Save Default Settings).

Once the datasets of user interest are in the table, the user can select what datasets are to be used as axes (first table column containing checkboxes), as signals (second column containing checkboxes) and eventually as monitor (third column with checkboxes). The only selection that is mandatory to generate a plot is the one corresponding to the signal.

In case of selecting several axes, the order in which the check boxes were selected determines the dataset to be used as first, second or third axis.

NeXus Support
-------------

`NeXus <http://www.nexusformat.org>`_ provides a set of directives to share data among different facilities. It provides an API supporting an HDF4 backend, an XML backend and an HDF5 backend. PyMca does not use the NeXus API and therefore only supports the HDF5 backend. On the other hand, HDF5 is the most common NeXus backend used at large scale research facilities.

NeXus HDF5 files can be handled in the same way as standard HDF5 files. In addition, PyMca will try to make as much use as possible of metadata, default plots and application definitions provided by NeXus to reduce user interaction.

For instance, version 5.3.0 of PyMca highlights NXdata groups in blue and a double-click on them allows direct visualization using the `silx library <https://www.silx.org>`_

Measurement Group Support
-------------------------

Some facilities follow what we can call the *measurement group approach* when collecting data. It is an additional convention to NeXus characterized by the addition of a group named *measurement* to each NXentry. The goal of that group is to provide the user with a quick access to information without the burden of having to hunt for the information in the highly hierarchical layout imposed by NeXus.

The *measurement* group was thought having in mind interactive handling of HDF5 files by users. Despite that, PyMca exploits that to provide an automatically filled selection table based on the contents of a *measurement* if present. Therefore, besides the USER selection table described above, PyMca provides the AUTO selection table automatically generated.

Other facilities follow a different approach consiting on having an NXdata group as container irrespectively of defining a default plot or not. In an attempt to offer the described functionality to users dealing with data from those facilities, PyMca also fills the AUTO table with the datasets found in the NXdata group containing the largest number of datasets.

This is implemented by the function :func:`PyMca5.PyMcaCore.NexusTools.getMeasurementGroup`

Positioners
-----------

PyMca tries to retrieve as much information as possible associated to the selections performed in the HDF5 files. In particular, and in analogy with what is available when dealing with `SPEC <https://www.certif.com/>`_ files, it tries to retrieve the information about the positioners (motors, temperatures, ...) associated to that selection. The conventions that PyMca is able to follow are:

- Presence of a group named *positioners*  inside an NXinstrument group (ESRF convention)
- Presence of a group named *pre_scan_snapshot* inside the measurement group (Sardana convention)

This is implemented by the function :func:`PyMca5.PyMcaCore.NexusTools.getPositionersGroup`

MCA Data
--------

When selecting a dataset as MCA, PyMca will try to retrieve associated information like the associated channels, live_time, elapsed_time, preset_time and calibration. For that to happen, datasets with those names should be present at the same level as the target dataset.

Please refer to the function :func:`PyMca5.PyMcaCore.NexusTools.getMcaObjectPaths` for details.


Similar to the AUTO table, PyMca tries to to build a selection table named MCA for datasets that may be considered as containing 1D data. For that, it searches for datasets containing the attribute *interpretation*  set to *spectrum*

Please refer to the function :func:`PyMca5.PyMcaCore.NexusTools.getMcaList` for details.

The following command::

   python -m PyMca5.PyMcaCore.NexusTools [your_HDF5_file_name]

will show you the information PyMca can automatically retrieve in terms of measurement groups, positioners, scanned motors, MCAs and associated information
