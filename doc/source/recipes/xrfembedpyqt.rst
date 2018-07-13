Embedding PyMca XRF fitting
===========================

Besides providing ready-to-use applications, PyMca is very modular and it allows to be used as a library.

Let's say you have your own way of displaying your data into a PyQt5 (or PyQt4, PySide or PySide2)
application. All you need to do to provide XRF fitting capabilities to it requires 4 lines of code.

.. code-block:: python

   from PyMca5.PyMca import McaAdvancedFit   
   widget = McaAdvancedFit.McaAdvancedFit()
   widget.setData(channels, counts)
   widget.show()

