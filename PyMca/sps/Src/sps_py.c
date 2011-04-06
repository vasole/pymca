/****************************************************************************
*
*   Copyright (c) 1998-2011 European Synchrotron Radiation Facility (ESRF)
*
*   The software contained in this file "sps_py.c" is designed to interface
*   the shared-data structures used and defined by the CSS "spec" package
*   with other utility software.
*
*   Permission is hereby granted, free of charge, to any person obtaining a
*   copy of the software in this file and associated header files (the
*   "Software"), to deal in the Software without restriction, including
*   without limitation the rights to use, copy, modify, merge, publish,
*   distribute, sublicense, and/or sell copies of the Software, and to
*   permit persons to whom the Software is furnished to do so, subject to
*   the following conditions:
*
*   The above copyright notice and this permission notice shall be included
*   in all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
*   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
*   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
*   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
****************************************************************************/
/* #include <stdlib.h> */
#include <sps.h>
/* #include <stdio.h> */
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *SPSError;

void        initsps(void);
static void sps_cleanup(void);

static int sps_type2py (int t)
{
  switch (t) {
  case SPS_USHORT: return(NPY_USHORT);
  case SPS_UINT:   return(NPY_UINT32);
  case SPS_UCHAR:  return(NPY_UBYTE);
  case SPS_SHORT:  return(NPY_SHORT);
  case SPS_INT:    return(NPY_INT32);
  case SPS_CHAR:   return(NPY_BYTE);
  case SPS_STRING: return(NPY_STRING);
  case SPS_FLOAT:  return(NPY_FLOAT);
  case SPS_DOUBLE: return(NPY_DOUBLE);
  default:        return(-1);
  }
}

static int sps_py2type (int t)
{
  int type;

  switch (t) {
  case NPY_INT32:
    type = SPS_INT; break;
  case NPY_UINT32:
    type = SPS_UINT; break;
  case NPY_USHORT:
    type = SPS_USHORT; break;
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
  case NPY_STRING:
    type = SPS_STRING; break;
  default:
    type = -1;
  }

  return type;
}

static PyObject * sps_getkeylist(PyObject *self, PyObject *args)
{
  char *spec_version=NULL, *array_name=NULL;
  int i;
  char *key;
  PyObject *list, *string;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  list = PyList_New(0);
  for (i=0; (key = SPS_GetNextEnvKey (spec_version,array_name, i)) ; i++) {
    string = PyString_FromString(key);
    PyList_Append (list, string);
    Py_DECREF(string);
  }

  return list;
}

static PyObject * sps_getarraylist(PyObject *self, PyObject *args)
{
  char *spec_version=NULL;
  int i;
  char *array;
  PyObject *list, *string;

  if (!PyArg_ParseTuple(args, "|s", &spec_version)) {
    return NULL;
  }

  list = PyList_New(0);
  for (i=0; (array = SPS_GetNextArray (spec_version,i)) ; i++) {
    string = PyString_FromString(array);
    PyList_Append (list, string);
    Py_DECREF(string);
  }

  return list;
}

static PyObject * sps_getspeclist(PyObject *self, PyObject *args)
{
  char *spec_version;
  int i;
  PyObject *list, *string;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  list = PyList_New(0);
  for (i=0; (spec_version = SPS_GetNextSpec (i)) ; i++) {
    string = PyString_FromString(spec_version);
    PyList_Append (list, string);
    Py_DECREF(string);
  }

  return list;
}

static PyObject *sps_isupdated(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  return PyInt_FromLong(SPS_IsUpdated(spec_version, array_name));
}

static PyObject *sps_updatecounter(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  return PyInt_FromLong(SPS_UpdateCounter(spec_version, array_name));
}

static PyObject *sps_updatedone(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  return PyInt_FromLong(SPS_UpdateDone(spec_version, array_name));
}

static PyObject *sps_getenvstr(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name, *key, *ret;

  if (!PyArg_ParseTuple(args, "sss", &spec_version, &array_name, &key)) {
    return NULL;
  }

  ret = SPS_GetEnvStr(spec_version, array_name, key);

  if (ret) {
    return PyString_FromString(ret);
  } else {
        PyErr_SetString(SPSError, "Key not found");
    return NULL;
  }
}

static PyObject *sps_putenvstr(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name, *key, *v;

  if (!PyArg_ParseTuple(args, "ssss", &spec_version, &array_name, &key, &v)) {
    return NULL;
  }

  if (SPS_PutEnvStr(spec_version, array_name, key, v)) {
    PyErr_SetString(SPSError, "Error setting the environment string");
    return NULL;
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject *sps_getarrayinfo(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag;


  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  return Py_BuildValue("(iiii)", rows, cols, type, flag);
}

static PyObject *sps_attach(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag;
  int dims[2];
  int ptype, stype;
  PyArrayObject *arrobj;
  void *data;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  if ((data = SPS_GetDataPointer(spec_version, array_name, 1)) == NULL) {
    PyErr_SetString(SPSError, "Error getting data pointer");
    return NULL;
  }

  dims[0]=rows;
  dims[1]=cols;
  ptype = sps_type2py(type);
  stype = sps_py2type(ptype);

  if (type != stype) {
    SPS_ReturnDataPointer(data);
    PyErr_SetString(SPSError, "Type of data in shared memory not supported");
    return NULL;
  }

  if ((arrobj = (PyArrayObject*) PyArray_FromDimsAndData(2, dims, ptype, data))
      == NULL) {
    SPS_ReturnDataPointer(data);
    PyErr_SetString(SPSError, "Could not create mathematical array");
    return NULL;
  }

  return (PyObject*) arrobj;
}

static PyObject *sps_detach(PyObject *self, PyObject *args)
{
  PyObject *in_arr;
  void *data;

  if (!PyArg_ParseTuple(args, "O", &in_arr)) {
    return NULL;
  }

  if (!PyArray_Check(in_arr)) {
    PyErr_SetString(SPSError, "Input must be the array returned by attach");
    return NULL;
  }

  data = ((PyArrayObject*) in_arr)->data;

  if (SPS_ReturnDataPointer(data)) {
    PyErr_SetString(SPSError, "Error detaching");
    return NULL;
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject *sps_create(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type = SPS_DOUBLE, flag = 0;
  int dims[2];
  int ptype, stype;
  PyArrayObject *arrobj;
  void *data;

  if (!PyArg_ParseTuple(args, "ssii|ii", &spec_version, &array_name,
            &rows, &cols, &type, &flag)) {
    return NULL;
  }

  if (SPS_CreateArray(spec_version, array_name, rows, cols, type, flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  if ((data = SPS_GetDataPointer(spec_version, array_name, 1)) == NULL) {
    PyErr_SetString(SPSError, "Error getting data pointer");
    return NULL;
  }

  dims[0]=rows;
  dims[1]=cols;
  ptype = sps_type2py(type);
  stype = sps_py2type(ptype);

  if (type != stype) {
    PyErr_SetString(SPSError, "Type of data in shared memory not supported");
    return NULL;
  }

  if ((arrobj = (PyArrayObject*) PyArray_FromDimsAndData(2, dims, ptype, data))
      == NULL) {
    /* Should delete the array  - don't have a lib function !!! FIXTHIS */
    PyErr_SetString(SPSError, "Could not create mathematical array");
    return NULL;
  }

  return (PyObject*) arrobj;
}

static PyObject *sps_getshmid(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag;
  int shmid;
  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
     return NULL;
   }
  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }
  shmid = SPS_GetShmId(spec_version, array_name);

  return Py_BuildValue("i", shmid);
}

static PyObject *sps_getdata(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag;
  npy_intp dims[2];
  int ptype, stype;
  PyArrayObject *arrobj, *arrobj_nc;

  if (!PyArg_ParseTuple(args, "ss", &spec_version, &array_name)) {
    return NULL;
  }

  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  dims[0]=rows;
  dims[1]=cols;
  ptype = sps_type2py(type);
  if ((arrobj_nc = (PyArrayObject*) PyArray_SimpleNew(2, dims, ptype))
      == NULL) {
    PyErr_SetString(SPSError, "Could not create mathematical array");
    return NULL;
  }

  if ((arrobj = (PyArrayObject*) PyArray_ContiguousFromObject(
           (PyObject*) arrobj_nc, ptype, 2, 2)) == NULL) {
    Py_DECREF(arrobj_nc);
    PyErr_SetString(SPSError, "Could not make our array contiguous");
    return NULL;
  } else
    Py_DECREF(arrobj_nc);

  stype = sps_py2type(ptype);
  SPS_CopyFromShared(spec_version, array_name, arrobj->data, stype ,
             rows * cols);

  return (PyObject*) arrobj;
}

static PyObject *sps_getdatarow(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag, in_row, in_col = 0;
  npy_intp dims[2];
  int ptype, stype;
  PyArrayObject *arrobj, *arrobj_nc;

  if (!PyArg_ParseTuple(args, "ssi|i", &spec_version, &array_name, &in_row,
            &in_col)) {
    return NULL;
  }

  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  dims[0] = (in_col == 0) ? cols : in_col;

  ptype = sps_type2py(type);
  if ((arrobj_nc = (PyArrayObject*) PyArray_SimpleNew(1, dims, ptype))
      == NULL) {
    PyErr_SetString(SPSError, "Could not create mathematical array");
    return NULL;
  }

  if ((arrobj = (PyArrayObject*) PyArray_ContiguousFromObject(
            (PyObject*) arrobj_nc, ptype, 1, 1)) == NULL) {
    Py_DECREF(arrobj_nc);
    PyErr_SetString(SPSError, "Could not make our array contiguous");
    return NULL;
  } else
    Py_DECREF(arrobj_nc);

  stype = sps_py2type(ptype);
  SPS_CopyRowFromShared(spec_version, array_name, arrobj->data, stype ,
             in_row, in_col, NULL);

  return (PyObject*) arrobj;
}

static PyObject *sps_getdatacol(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int rows, cols, type, flag, in_row = 0, in_col;
  int dims[2];
  int ptype, stype;
  PyArrayObject *arrobj, *arrobj_nc;

  if (!PyArg_ParseTuple(args, "ssi|i", &spec_version, &array_name, &in_col,
            &in_row)) {
    return NULL;
  }

  if (SPS_GetArrayInfo(spec_version, array_name, &rows, &cols, &type, &flag)) {
    PyErr_SetString(SPSError, "Error getting array info");
    return NULL;
  }

  dims[0] = (in_row == 0) ? rows : in_row;

  ptype = sps_type2py(type);
  if ((arrobj_nc = (PyArrayObject*) PyArray_FromDims(1, dims, ptype))
      == NULL) {
    PyErr_SetString(SPSError, "Could not create mathematical array");
    return NULL;
  }

  if ((arrobj = (PyArrayObject*) PyArray_ContiguousFromObject(
               (PyObject*) arrobj_nc, ptype, 1, 1)) == NULL) {
    Py_DECREF(arrobj_nc);
    PyErr_SetString(SPSError, "Could not make our array contiguous");
    return NULL;
  } else
    Py_DECREF(arrobj_nc);

  stype = sps_py2type(ptype);
  SPS_CopyColFromShared(spec_version, array_name, arrobj->data, stype ,
             in_col, in_row, NULL);
  return (PyObject*) arrobj;
}

static PyObject *sps_putdata(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int ptype, stype;
  PyObject *in_src;
  PyArrayObject *src;
  int no_items;

  if (!PyArg_ParseTuple(args, "ssO", &spec_version, &array_name, &in_src)) {
    return NULL;
  }

  if (!(src = (PyArrayObject*) PyArray_ContiguousFromObject(in_src,
                PyArray_NOTYPE, 2, 2))) {
    PyErr_SetString(SPSError, "Input Array is not a 2 dim array");
    return NULL;
  }

  ptype = src->descr->type_num;
  stype = sps_py2type(ptype);

  if (ptype != sps_type2py(stype)) {
    PyErr_SetString(SPSError, "Type of data in shared memory not supported");
    Py_DECREF(src);
    return NULL;
  }

  no_items = src->dimensions[0] * src->dimensions[1];

  if (SPS_CopyToShared(spec_version, array_name, src->data, stype, no_items)
      == -1) {
    PyErr_SetString(SPSError, "Error copying data to shared memory");
    Py_DECREF(src);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *sps_putdatarow(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int ptype, stype;
  PyObject *in_src;
  PyArrayObject *src;
  int no_items;
  int in_row;


  if (!PyArg_ParseTuple(args, "ssiO", &spec_version, &array_name, &in_row,
            &in_src)) {
    return NULL;
  }

  if (!(src = (PyArrayObject*) PyArray_ContiguousFromObject(in_src,
                PyArray_NOTYPE, 1, 1))) {
    PyErr_SetString(SPSError, "Input Array is not a 1 dim array");
    return NULL;
  }

  ptype = src->descr->type_num;
  stype = sps_py2type(ptype);

  if (ptype == -1) {
    PyErr_SetString(SPSError, "Type of data in shared memory not supported");
    Py_DECREF(src);
    return NULL;
  }

  no_items = src->dimensions[0];

  if (SPS_CopyRowToShared(spec_version, array_name, src->data, stype,
              in_row, no_items, NULL)
      == -1) {
    PyErr_SetString(SPSError, "Error copying data to shared memory");
    Py_DECREF(src);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *sps_putdatacol(PyObject *self, PyObject *args)
{
  char *spec_version, *array_name;
  int ptype, stype;
  PyObject *in_src;
  PyArrayObject *src;
  int no_items;
  int in_col = 0;


  if (!PyArg_ParseTuple(args, "ssiO", &spec_version, &array_name, &in_col,
            &in_src)) {
    return NULL;
  }

  if (!(src = (PyArrayObject*) PyArray_ContiguousFromObject(in_src,
                PyArray_NOTYPE, 1, 1))) {
    PyErr_SetString(SPSError, "Input Array is not a 1 dim array");
    return NULL;
  }

  ptype = src->descr->type_num;
  stype = sps_py2type(ptype);

  no_items = src->dimensions[0];

  if (SPS_CopyColToShared(spec_version, array_name, src->data, stype,
              in_col, no_items, NULL)
      == -1) {
    PyErr_SetString(SPSError, "Error copying data to shared memory");
    Py_DECREF(src);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static void sps_cleanup()
{
  SPS_CleanUpAll();
}

static PyMethodDef SPSMethods[] = {
  { "getspeclist",   sps_getspeclist,METH_VARARGS},
  { "getarraylist",  sps_getarraylist, METH_VARARGS},
  { "isupdated",     sps_isupdated,  METH_VARARGS},
  { "updatecounter", sps_updatecounter, METH_VARARGS},
  { "getenv",        sps_getenvstr,  METH_VARARGS},
  { "putenv",        sps_putenvstr,  METH_VARARGS},
  { "getkeylist",    sps_getkeylist, METH_VARARGS},
  { "getshmid",      sps_getshmid,   METH_VARARGS},
  { "getdata",       sps_getdata,    METH_VARARGS},
  { "getdatarow",    sps_getdatarow, METH_VARARGS},
  { "getdatacol",    sps_getdatacol, METH_VARARGS},
  { "getarrayinfo",  sps_getarrayinfo, METH_VARARGS},
  { "attach",        sps_attach,     METH_VARARGS},
  { "detach",        sps_detach,     METH_VARARGS},
  { "create",        sps_create,     METH_VARARGS},
  { "updatedone",    sps_updatedone, METH_VARARGS},
  { "putdata",       sps_putdata,    METH_VARARGS},
  { "putdatarow",    sps_putdatarow, METH_VARARGS},
  { "putdatacol",    sps_putdatacol, METH_VARARGS},
  { NULL, NULL}
};

void initsps()
{
  PyObject *d, *m;
  m = Py_InitModule ("sps", SPSMethods);

  //printf("Initializing sps\n");
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "DOUBLE",   PyInt_FromLong(SPS_DOUBLE));
  PyDict_SetItemString(d, "FLOAT",    PyInt_FromLong(SPS_FLOAT));
  PyDict_SetItemString(d, "INT",      PyInt_FromLong(SPS_INT));
  PyDict_SetItemString(d, "UINT",     PyInt_FromLong(SPS_UINT));
  PyDict_SetItemString(d, "SHORT",    PyInt_FromLong(SPS_SHORT));
  PyDict_SetItemString(d, "USHORT",   PyInt_FromLong(SPS_USHORT));
  PyDict_SetItemString(d, "CHAR",     PyInt_FromLong(SPS_CHAR));
  PyDict_SetItemString(d, "UCHAR",    PyInt_FromLong(SPS_UCHAR));
  PyDict_SetItemString(d, "STRING",   PyInt_FromLong(SPS_STRING));

  PyDict_SetItemString(d, "IS_ARRAY", PyInt_FromLong(SPS_IS_ARRAY));
  PyDict_SetItemString(d, "IS_MCA",   PyInt_FromLong(SPS_IS_MCA));
  PyDict_SetItemString(d, "IS_IMAGE", PyInt_FromLong(SPS_IS_IMAGE));

  PyDict_SetItemString(d, "TAG_STATUS", PyInt_FromLong(SPS_TAG_STATUS));
  PyDict_SetItemString(d, "TAG_ARRAY", PyInt_FromLong(SPS_TAG_ARRAY));
  PyDict_SetItemString(d, "TAG_MASK", PyInt_FromLong(SPS_TAG_MASK));
  PyDict_SetItemString(d, "TAG_MCA", PyInt_FromLong(SPS_TAG_MCA));
  PyDict_SetItemString(d, "TAG_IMAGE", PyInt_FromLong(SPS_TAG_IMAGE));
  PyDict_SetItemString(d, "TAG_SCAN", PyInt_FromLong(SPS_TAG_SCAN));
  PyDict_SetItemString(d, "TAG_INFO", PyInt_FromLong(SPS_TAG_INFO));
  PyDict_SetItemString(d, "TAG_FRAMES", PyInt_FromLong(SPS_TAG_FRAMES));


  SPSError = PyErr_NewException("sps.error", NULL, NULL);
  PyDict_SetItemString(d, "error", SPSError);

  Py_AtExit(sps_cleanup);
  import_array();
}
