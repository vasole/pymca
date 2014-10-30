#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
/* spslut_py.c VERSION 4.0 */
/* $Revision: 1.7 $
 * $Log: spslut_py.c,v $
 * Revision 1.7  2005/02/10 23:37:48  sole
 * minor changes
 *
 * Revision 1.6  2005/02/10 17:44:58  sole
 * *** empty log message ***
 *
 * Revision 1.5  2005/02/10 16:17:15  sole
 * Removed some unused variables
 **/
/* CHANGES:

    [05-09-2002] A. Gobbo
    - Included min and max values to 8 bit colormaps

    [11-03-2002] A. Gobbo
    - Included modes BGR and BGRX

    [12-12-2001] A. Gobbo
    - Dimentions inverted in the returned array
    - Corrected memory leak bug
*/
#include <stdio.h>
#include <Python.h>
/*
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
*/
#include <numpy/arrayobject.h>
#include <sps_lut.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define GETSTATE(m) ((struct module_state*) PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define onError(message)  \
     {	struct module_state *st = GETSTATE(self);\
		PyErr_SetString(st->error, message);\
		return NULL; }

/* Function declarations */
static PyObject *spslut_transform(PyObject *self, PyObject *args);
static PyObject *spslut_transformarray(PyObject *self, PyObject *args);
static PyObject *spslut_palette(PyObject *self, PyObject *args);

/* Functions */

PyObject *new_pyimage(const char *mode, unsigned xsize, unsigned ysize,
              void *data)
{
#if PY_MAJOR_VERSION >= 3
  return PyBytes_FromStringAndSize ((const char *)data,
                     strlen(mode) * xsize * ysize);
#else
  return PyString_FromStringAndSize ((const char *)data,
                     strlen(mode) * xsize * ysize);
#endif
}

static int natbyteorder()
{
  union {
    struct {
      unsigned char b1;
      unsigned char b2;
      unsigned char b3;
      unsigned char b4;
    } c;
    unsigned long p;
  } val;

  val.p = 1;
  if (val.c.b4 == 1) {
    return SPS_MSB;
  } else {
    return SPS_LSB;
  }
}

static PyObject *spslut_transform(PyObject *self, PyObject *args)
{
  void *data;
  int type, cols, rows, reduc, fastreduc, meth, autoscale, mapmin=0, mapmax=255;
  int palette_code;
  double gamma, min, max;
  XServer_Info Xservinfo;
  void *palette;
  int prows, pcols, pal_entries;
  void *r/*, *res*/;
  char *mode;
  PyArrayObject *src;
  PyObject *in_src;
  PyObject *res,*aux;
  int array_output=0;
  unsigned char *as_pointer, *as_r;
  npy_intp as_dim[2];
  PyArrayObject *as_aux;

  if (!PyArg_ParseTuple(args, "O(ii)(id)sii(dd)|(ii)i", &in_src, &reduc,
            &fastreduc, &meth, &gamma, &mode, &palette_code,
            &autoscale, &min, &max,&mapmin, &mapmax, &array_output))
        return NULL;

  if (strcmp(mode, "RGB") == 0) {
    Xservinfo.red_mask = 0x0000ff;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0xff0000;
    Xservinfo.pixel_size = 3;
    Xservinfo.byte_order = natbyteorder();
  } else if (strcmp(mode, "RGBX") == 0) {
    Xservinfo.red_mask = 0x0000ff;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0xff0000;
    Xservinfo.pixel_size = 4;
    Xservinfo.byte_order = natbyteorder();
  }

  //###CHANGED - ALEXANDRE 11/03/2002 - Qt uses different order than Tkinter
  else if (strcmp(mode, "BGR") == 0) {
    Xservinfo.red_mask = 0xff0000;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0x0000ff;
    Xservinfo.pixel_size = 3;
    Xservinfo.byte_order = natbyteorder();
  } else if (strcmp(mode, "BGRX") == 0) {
    Xservinfo.red_mask = 0xff0000;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0x0000ff;
    Xservinfo.pixel_size = 4;
    Xservinfo.byte_order = natbyteorder();
  }

    else if (strcmp(mode, "L") == 0 || strcmp(mode, "P") == 0  ) {
    Xservinfo.pixel_size = 1;
    Xservinfo.byte_order = natbyteorder();
    //mapmin = 0;
    //mapmax = 255;
  } else {
    onError("Mode must be RGB, RGBX, BGR, BGRX, L or P");
  }

  if (!(src = (PyArrayObject*) PyArray_ContiguousFromObject(in_src,
                NPY_NOTYPE, 2, 2))) {
    onError("Input Array is not a 2x2 array");
  }

  switch (PyArray_DESCR(src)->type_num) {
  case NPY_UINT:
    type = SPS_UINT; break;
  case NPY_ULONG:
    type = SPS_ULONG; break;
  case NPY_USHORT:
    type = SPS_USHORT; break;
  case NPY_LONG:
    type = SPS_LONG; break;
  case NPY_INT:
    type = SPS_INT; break;
  case NPY_SHORT:
    type = SPS_SHORT; break;
  case NPY_UBYTE:
    type = SPS_UCHAR; break;
  case NPY_BYTE:
    type = SPS_CHAR; break;
  case NPY_FLOAT:
    type = SPS_FLOAT; break;
  case NPY_DOUBLE:
    type = SPS_DOUBLE; break;
  default:
    onError("Input Array type not supported");
  }

  data = PyArray_DATA(src);
  cols = (int) PyArray_DIMS(src)[1];  /*FIX THIS cols and rows are turned around */
  rows = (int) PyArray_DIMS(src)[0];  /*###CHANGED - ALEXANDRE 24/07/2001*/



  r = SPS_PaletteArray (data, type, cols, rows, reduc, fastreduc, meth, gamma,
            autoscale, mapmin, mapmax, Xservinfo, palette_code,
            &min, &max, &pcols, &prows, &palette, &pal_entries);
  if (r == 0) {
    onError("Error while trying to calculate the image");
  }
  if (!array_output){
   /*###CHANGED - ALEXANDRE 24/07/2001*/
  aux=new_pyimage(mode, (unsigned) pcols, (unsigned) prows, r);
  res = Py_BuildValue("(O(i,i)(d,d))",aux,pcols, prows, min, max);
  free(r);
  Py_DECREF(aux);

  /*###CHANGED - ALEXANDRE 28/06/2002*/
  Py_DECREF(src);


  return res;
  }
  as_dim[0] = strlen(mode);
  as_dim[1] = prows * pcols;
  as_aux = (PyArrayObject*) PyArray_SimpleNew(2, as_dim, NPY_UBYTE);
  if (as_aux == NULL){
      free(r);
      Py_DECREF(src);
      return NULL;
  }
  as_pointer = (char *) PyArray_DATA(as_aux);
  as_r = (char *) r;

  memcpy(as_pointer, as_r, as_dim[0] * as_dim[1]);
  free(r);
  res = Py_BuildValue("(O(i,i)(d,d))",as_aux,pcols, prows, min, max);
  Py_DECREF(src);
  Py_DECREF(as_aux);
  return res;
}


static PyObject *spslut_transformarray(PyObject *self, PyObject *args)
{
  void *data;
  int type, cols, rows, reduc, fastreduc, meth, autoscale, mapmin=0, mapmax=255;
  int palette_code;
  double gamma, min, max;
  XServer_Info Xservinfo;
  void *palette;
  int prows, pcols, pal_entries;
  void *r/*, *res*/;
  char *mode;
  unsigned char *as_pointer, *as_r;
  npy_intp as_dim[3];
  PyArrayObject *src;
  PyObject *in_src;
  PyArrayObject *aux;

  if (!PyArg_ParseTuple(args, "O(ii)(id)sii(dd)|(ii)", &in_src, &reduc,
            &fastreduc, &meth, &gamma, &mode, &palette_code,
            &autoscale, &min, &max,&mapmin, &mapmax))
        return NULL;

  if (strcmp(mode, "RGB") == 0) {
    Xservinfo.red_mask = 0x0000ff;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0xff0000;
    Xservinfo.pixel_size = 3;
    Xservinfo.byte_order = natbyteorder();
  } else if (strcmp(mode, "RGBX") == 0) {
    Xservinfo.red_mask = 0x0000ff;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0xff0000;
    Xservinfo.pixel_size = 4;
    Xservinfo.byte_order = natbyteorder();
  }

  //###CHANGED - ALEXANDRE 11/03/2002 - Qt uses different order than Tkinter
  else if (strcmp(mode, "BGR") == 0) {
    Xservinfo.red_mask = 0xff0000;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0x0000ff;
    Xservinfo.pixel_size = 3;
    Xservinfo.byte_order = natbyteorder();
  } else if (strcmp(mode, "BGRX") == 0) {
    Xservinfo.red_mask = 0xff0000;
    Xservinfo.green_mask = 0x00ff00;
    Xservinfo.blue_mask = 0x0000ff;
    Xservinfo.pixel_size = 4;
    Xservinfo.byte_order = natbyteorder();
  }

    else if (strcmp(mode, "L") == 0 || strcmp(mode, "P") == 0  ) {
    Xservinfo.pixel_size = 1;
    Xservinfo.byte_order = natbyteorder();
    //mapmin = 0;
    //mapmax = 255;
  } else {
    onError("Mode must be RGB, RGBX, BGR, BGRX, L or P");
  }

  if (!(src = (PyArrayObject*) PyArray_ContiguousFromObject(in_src,
                NPY_NOTYPE, 2, 2))) {
		onError("spslut.transformarray: Input Array is not a 2x2 array");
  }

  switch (PyArray_DESCR(src)->type_num) {
  case NPY_ULONG:
    type = SPS_ULONG; break;
  case NPY_UINT:
    type = SPS_UINT; break;
  case NPY_USHORT:
    type = SPS_USHORT; break;
  case NPY_LONG:
    type = SPS_LONG; break;
  case NPY_INT:
    type = SPS_INT; break;
  case NPY_SHORT:
    type = SPS_SHORT; break;
  case NPY_UBYTE:
    type = SPS_UCHAR; break;
  case NPY_BYTE:
    type = SPS_CHAR; break;
  case NPY_FLOAT:
    type = SPS_FLOAT; break;
  case NPY_DOUBLE:
    type = SPS_DOUBLE; break;
  default:
    onError("Input Array type not supported");
  }

  data = PyArray_DATA(src);
  cols = (int) PyArray_DIMS(src)[1];  /*FIX THIS cols and rows are turned around */
  rows = (int) PyArray_DIMS(src)[0];  /*###CHANGED - ALEXANDRE 24/07/2001*/
  r = SPS_PaletteArray (data, type, cols, rows, reduc, fastreduc, meth, gamma,
            autoscale, mapmin, mapmax, Xservinfo, palette_code,
            &min, &max, &pcols, &prows, &palette, &pal_entries);
  if (r == 0) {
    onError("Error while trying to calculate the image");
  }
   /*###CHANGED - ALEXANDRE 24/07/2001*/
  as_dim[0] = strlen(mode);
  as_dim[1] = prows * pcols;
/*  printf("dim[0] = %d dim[1] = %d \"%s\" \n",as_dim[0],as_dim[1],mode); */
  aux = (PyArrayObject*) PyArray_SimpleNew(2, as_dim, NPY_UBYTE);
  if (aux == NULL){
      free(r);
      Py_DECREF(src);
      return NULL;
  }
  as_pointer = (char *) PyArray_DATA(aux);
  as_r = (char *) r;
  memcpy(as_pointer, as_r, as_dim[0] * as_dim[1]);
  free(r);
  Py_DECREF(src);
  return PyArray_Return(aux);
}



/* The simple palette always returns 4 bytes per entry */
static PyObject *spslut_palette(PyObject *self, PyObject *args)
{
  int entries, palette_code;
  XServer_Info Xservinfo;
  void *r;
  char *mode;

  if (!PyArg_ParseTuple(args, "ii", &entries, &palette_code))
    return NULL;

  mode = "RGBX";
  Xservinfo.red_mask = 0x0000ff;
  Xservinfo.green_mask = 0x00ff00;
  Xservinfo.blue_mask = 0xff0000;
  Xservinfo.pixel_size = 4;
  Xservinfo.byte_order = natbyteorder();

  r = SPS_SimplePalette ( 0, entries - 1, Xservinfo, palette_code);

  if (r == 0) {
    onError("Error calculating the palette");
  }
  return new_pyimage(mode, 1, entries, r);
}


/* Module methods */

static PyMethodDef SPSLUT_Methods[] = {
  { "transform", spslut_transform, METH_VARARGS},
  { "palette", spslut_palette, METH_VARARGS},
  { "transformArray", spslut_transformarray, METH_VARARGS},
  { NULL, NULL}
};

/* ------------------------------------------------------- */

/* Module initialization */

#if PY_MAJOR_VERSION >= 3

static int SPSLUT_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int SPSLUT_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spslut",
        NULL,
        sizeof(struct module_state),
        SPSLUT_Methods,
        NULL,
        SPSLUT_traverse,
        SPSLUT_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_spslut(void)

#else
#define INITERROR return

void
initspslut(void)
#endif
{
	PyObject *d;
	struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("spslut", SPSLUT_Methods);
#endif

    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("SPSLUT.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    import_array();
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);

  PyDict_SetItemString(d, "LINEAR", PyInt_FromLong(SPS_LINEAR));
  PyDict_SetItemString(d, "LOG", PyInt_FromLong(SPS_LOG));
  PyDict_SetItemString(d, "GAMMA", PyInt_FromLong(SPS_GAMMA));

  PyDict_SetItemString(d, "GREYSCALE", PyInt_FromLong(SPS_GREYSCALE));
  PyDict_SetItemString(d, "TEMP", PyInt_FromLong(SPS_TEMP));
  PyDict_SetItemString(d, "RED", PyInt_FromLong(SPS_RED));
  PyDict_SetItemString(d, "GREEN", PyInt_FromLong(SPS_GREEN));
  PyDict_SetItemString(d, "BLUE", PyInt_FromLong(SPS_BLUE));
  PyDict_SetItemString(d, "REVERSEGREY", PyInt_FromLong(SPS_REVERSEGREY));
  PyDict_SetItemString(d, "MANY", PyInt_FromLong(SPS_MANY));

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
