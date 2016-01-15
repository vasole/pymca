#/*##########################################################################
# Copyright (C) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
#include <Python.h>
/* adding next line may raise errors ...
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
*/
#include <numpy/arrayobject.h>
#include <SpecFile.h>

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    SpecFile *sf;
    char     *name;
    short     length;
} specfileobject;

typedef struct {
   PyObject_HEAD
   specfileobject *file;
   long            index;
   long            cols;
} scandataobject;

static PyObject *SpecfileError;
#define onError(message)  \
     {PyErr_SetString (SpecfileError, message); return NULL; }

/*---------------------------*/

   /*
    * Utility function
    */

static char           * compList(long *nolist,long howmany);

static PyObject * specfile_close(PyObject *self);               /* dealloc */
static Py_ssize_t specfile_noscans(PyObject *self);             /* length  */
static PyObject * specfile_scan   (PyObject *self, Py_ssize_t index);   /* item    */
static int        specfile_print(PyObject *self,FILE *fp,
                                                   int flags);  /* print*/

   /*
    * Specfile methods
    */

static PyObject  * specfile_list      (PyObject *self,PyObject *args);
static PyObject  * specfile_allmotors (PyObject *self,PyObject *args);
static PyObject  * specfile_title     (PyObject *self,PyObject *args);
static PyObject  * specfile_user      (PyObject *self,PyObject *args);
static PyObject  * specfile_date      (PyObject *self,PyObject *args);
static PyObject  * specfile_epoch     (PyObject *self,PyObject *args);
static PyObject  * specfile_update    (PyObject *self,PyObject *args);
static PyObject  * specfile_scanno    (PyObject *self,PyObject *args);
static PyObject  * specfile_select    (PyObject *self,PyObject *args);
static PyObject  * specfile_show      (PyObject *self,PyObject *args);

static PyObject *
specfile_list(PyObject *self,PyObject *args)
{
    long      *scanlist;
    long      no_scans;
    int       error = 0;
    char     *strlist;
    PyObject *pstr;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    no_scans = SfScanNo(v->sf);
    scanlist = SfList(v->sf,&error);

    if ( scanlist == NULL || no_scans == 0) {
        PyErr_SetString(PyExc_TypeError,"Cannot get scan list for file");
        return NULL;
    } else {
        strlist = (char *)compList(scanlist,no_scans);
        pstr = Py_BuildValue("s",strlist);
        free(scanlist);
        return pstr;
    }
}

static PyObject *
specfile_allmotors(PyObject *self,PyObject *args)
{
    int error,i;
    char **motornames;
    long nb_motors;
    PyObject *t,*x;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    nb_motors = SfAllMotors(v->sf,1,&motornames,&error);

    if ( nb_motors == -1 )
           onError("cannot get motor names for specfile");

    t = PyList_New(nb_motors);
    for ( i = 0 ;i<nb_motors;i++) {
       x = PyUnicode_FromString(motornames[i]);
       PyList_SetItem(t,i,x);
    }

    return t;
}

static PyObject *
specfile_title(PyObject *self,PyObject *args)
{
    int error;
    char *title;
    PyObject *pyo;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    title = SfTitle(v->sf,1,&error);

    if (title == NULL)
        onError("cannot get title for specfile")

    pyo = Py_BuildValue("s",title);
    free(title);
    return pyo;
}

static PyObject *
specfile_user(PyObject *self,PyObject *args)
{
    int error;
    char *user;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    user = SfUser(v->sf,1,&error);

    if (user != NULL) {
        free(user);
        return Py_BuildValue("s",user);
    } else {
        onError("cannot get user for specfile");
    }
}

static PyObject *
specfile_date(PyObject *self,PyObject *args)
{
    int error;
    char *date;
    PyObject *pyo;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    date = SfFileDate(v->sf,1,&error);

    if (date == NULL)
       onError("cannot get data for specfile")

    pyo = Py_BuildValue("s",date);
    free(date);
    return pyo;
}

static PyObject *
specfile_epoch(PyObject *self,PyObject *args)
{
    int error;
    long epoch;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    epoch = SfEpoch(v->sf,1,&error);

    if (epoch != -1) {
        return Py_BuildValue("l",epoch);
    } else {
        onError("cannot get epoch for specfile");
    }
}

static PyObject *
specfile_update(PyObject *self,PyObject *args)
{
    int error;
    short ret;

    specfileobject *v = (specfileobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    ret = SfUpdate(v->sf,&error);
    if (ret == 1){
       v->length = SfScanNo(v->sf);
    }
    return(Py_BuildValue("i",ret));
}

static PyObject *
specfile_scanno(PyObject *self,PyObject *args)
{
    long scanno;
    specfileobject *v = (specfileobject *) self;


    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    scanno = v->length;

    return Py_BuildValue("l",scanno);
}

static PyObject *
specfile_show(PyObject *self,PyObject *args)
{

    specfileobject *f = (specfileobject *)self;

    SfShow(f->sf);

    return (Py_BuildValue("l",0));
}

static struct PyMethodDef  specfile_methods[] =
{
   {"list",      specfile_list,      1},
   {"allmotors", specfile_allmotors, 1},
   {"title",     specfile_title,     1},
   {"user",      specfile_user,      1},
   {"date",      specfile_date,      1},
   {"epoch",     specfile_epoch,     1},
   {"update",    specfile_update,    1},
   {"scanno",    specfile_scanno,    1},
   {"select",    specfile_select,    1},
   {"show",      specfile_show,      1},
   { NULL, NULL}
};

static PySequenceMethods specfile_as_sequence = {
  specfile_noscans,    /*     length len(sf)     */
  0,              /*     concat sf1 + sf2   */
  0,              /*     repeat sf * n      */
  specfile_scan,       /*     item  sf[i], in    */
  0,              /*     slice sf[i:j]      */
  0,              /*     asset sf[i] = v    */
  0,              /* slice ass. sf[i:j] = v */
};

static PyTypeObject Specfiletype = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "specfile",                /* tp_name */
    sizeof(specfileobject),    /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)specfile_close,/* tp_dealloc */
    (printfunc)specfile_print, /* tp-print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    &specfile_as_sequence,     /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "specfile objects",           /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    specfile_methods,             /* tp_methods */
    0, /*specfile_members,*/      /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0, /*(initproc)specfile_init,*/  /* tp_init */
    0,                         /* tp_alloc */
    0, /*specfile_new,0/          /* tp_new */
};

/*---------------------------*/
static PyObject *
specfile_open(PyObject *self, PyObject *args) /* on x = specfile.Specfile(name) */
{
    PyObject *input;
#ifdef WIN32
    PyObject *bytesObject;
#endif
    const char *filename;
    specfileobject *object;
    SpecFile       *sf;
    int             error;
    int                fd;

    /*
    if (!PyArg_ParseTuple(args, "s", &filename))
      return NULL;
     */

#ifdef WIN32
    if (!PyArg_ParseTuple(args, "O", &input))
    {
      return NULL;
    }
    {
        bytesObject = PyUnicode_AsMBCSString(input);
        if (!bytesObject){
            onError("Cannot generate String from object name attribute")
        }
        filename = PyBytes_AsString(bytesObject);
    }
#else
    if (!PyArg_ParseTuple(args, "s", &filename))
    {
        return NULL;
    }
#endif

    Specfiletype.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Specfiletype) < 0)
        return NULL;

    object = (specfileobject *) _PyObject_New(&Specfiletype);
    if (object == NULL)
        return NULL;
    object->sf = NULL;
    object->name = (char *)strdup(filename);
    strcpy(object->name, filename);
#ifdef WIN32
    Py_DECREF(bytesObject);
#endif
    if (( sf = SfOpen((char *) filename, &error)) == NULL )
    {
        Py_DECREF(object);
        onError("cannot open file");
    }
    object->sf = sf;
    object->length = (short) SfScanNo(sf);
    return (PyObject *) object;
}

static PyObject *
specfile_close(PyObject *self)
{
    specfileobject *f;
    f = (specfileobject *) self;
    if (f->sf)
    {
        SfClose(f->sf);
    }
    free(f->name);
    PyObject_DEL(f);
    return NULL;
}

  /*
   * Sequence type methods
   */
static Py_ssize_t
specfile_noscans(PyObject *self) {
    int       no_scans;
    specfileobject *f;
    f = (specfileobject *) self;

    no_scans = f->length;

    return (Py_ssize_t) no_scans;
}

static int
specfile_print(PyObject *self, FILE *fp, int flags)
{
    int ok=0;
    specfileobject *f;
    printf("Called\n");
    f = (specfileobject *)self;
    fprintf(fp, "specfile('%s')", f->name);
    return ok;
}

static PyObject *
specfile_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

/* end specfile class */
   /*
    * Scandata methods
    */

static PyObject   * scandata_data         (PyObject *self,PyObject *args);
static PyObject   * scandata_dataline     (PyObject *self,PyObject *args);
static PyObject   * scandata_datacol      (PyObject *self,PyObject *args);
static PyObject   * scandata_alllabels    (PyObject *self,PyObject *args);
static PyObject   * scandata_allmotors    (PyObject *self,PyObject *args);
static PyObject   * scandata_allmotorpos  (PyObject *self,PyObject *args);
static PyObject   * scandata_motorpos     (PyObject *self,PyObject *args);
static PyObject   * scandata_hkl          (PyObject *self,PyObject *args);
static PyObject   * scandata_number       (PyObject *self,PyObject *args);
static PyObject   * scandata_order        (PyObject *self,PyObject *args);
static PyObject   * scandata_command      (PyObject *self,PyObject *args);
static PyObject   * scandata_date         (PyObject *self,PyObject *args);
static PyObject   * scandata_cols         (PyObject *self,PyObject *args);
static PyObject   * scandata_lines        (PyObject *self,PyObject *args);
static PyObject   * scandata_header       (PyObject *self,PyObject *args);
static PyObject   * scandata_fileheader   (PyObject *self,PyObject *args);
static PyObject   * scandata_nbmca        (PyObject *self,PyObject *args);
static PyObject   * scandata_mca          (PyObject *self,PyObject *args);
static PyObject   * scandata_show         (PyObject *self,PyObject *args);

static struct PyMethodDef  scandata_methods[] = {
   {"data",        scandata_data,        1},
   {"dataline",    scandata_dataline,    1},
   {"datacol",     scandata_datacol,     1},
   {"alllabels",   scandata_alllabels,   1},
   {"allmotors",   scandata_allmotors,   1},
   {"allmotorpos", scandata_allmotorpos, 1},
   {"motorpos",    scandata_motorpos,    1},
   {"hkl",         scandata_hkl,         1},
   {"number",      scandata_number,      1},
   {"order",       scandata_order,       1},
   {"command",     scandata_command,     1},
   {"date",        scandata_date,        1},
   {"cols",        scandata_cols,        1},
   {"lines",       scandata_lines,       1},
   {"header",      scandata_header,      1},
   {"fileheader",  scandata_fileheader,  1},
   {"nbmca",       scandata_nbmca,       1},
   {"mca",         scandata_mca,         1},
   {"show",        scandata_show,        1},
   { NULL, NULL}
};

   /*
    * Scandata python basic operation
    */
static PyObject * scandata_new    (void);                                             /* create */
static PyObject * scandata_free   (PyObject *self);                                   /* dealloc */
static Py_ssize_t scandata_size   (PyObject *self);                                   /* length */
static PyObject * scandata_col    (PyObject *self, Py_ssize_t index);                  /* item */
static PyObject * scandata_slice  (PyObject *self, Py_ssize_t lidx, Py_ssize_t hidx); /* slice */
static int        scandata_print  (PyObject *self,FILE *fp, int flags);               /* print*/
static PyObject * scandata_getattr(PyObject *self,char *name);                        /* getattr */

static Py_ssize_t
scandata_size(PyObject *self) {

    scandataobject *s = (scandataobject *) self;

    return (Py_ssize_t) s->cols;
}

static PyObject *
scandata_col(PyObject *self, Py_ssize_t index) {
    int     error;
    npy_intp ret;
    double  *data;

    PyArrayObject *r_array;

    SpecFile *sf;
    int      idx,col;

    scandataobject *s = (scandataobject *) self;


    if ( index < 0 || index > (s->cols - 1) ) {
         PyErr_SetString(PyExc_IndexError,"column out of bounds");
         return NULL;
    }

    sf  = (s->file)->sf;
    idx = s->index;
    col = (int) (index + 1);

    ret = SfDataCol(sf,idx,col,&data,&error);

    if (ret == -1 )
          onError("cannot get data for column");

    r_array = (PyArrayObject *)PyArray_SimpleNew(1,&ret,NPY_DOUBLE);

    if ( r_array == NULL )
          onError("cannot get memory for array data");

    if (data != (double *) NULL){
        memcpy(PyArray_DATA(r_array), data, PyArray_NBYTES(r_array));
        free(data);
    }else{
        /* return an empty array? */
        printf("I should return an empty array ...\n");
        PyArray_FILLWBYTE(r_array, 0);
    }

/*    return (PyObject *)array; */
/* put back the PyArray_Return call instead of the previous line */
    return PyArray_Return(r_array);

    /*
      it does not work for solaris and linux
      I should check the call to PyErr_Occurred()) in Numeric/Src/arrayobject.c
      PyArray_Return(array);
     */
}

static PyObject *
scandata_slice(PyObject *self, Py_ssize_t ilow, Py_ssize_t ihigh) {
    return NULL;
}

static PyObject *
scandata_data(PyObject *self,PyObject *args) {

    int     error;
    int     ret;
    double  **data;
    long    *data_info;
    int i,j;
    npy_intp dimensions[2];

    SpecFile *sf;
    int     idx,didx;
    PyArrayObject *r_array;

    scandataobject *s = (scandataobject *) self;

    sf  = (s->file)->sf;
    idx = s->index;

    if (!PyArg_ParseTuple(args,"") )
           onError("wrong arguments for data");

    ret = SfData(sf,idx,&data,&data_info,&error);
    if ( ret == -1 )
           onError("cannot read data");

/*    printf("DATA: %d rows / %d columns\n", data_info[1], data_info[0]);*/

    dimensions[0] = data_info[1];
    dimensions[1] = data_info[0];
    r_array = (PyArrayObject *)PyArray_SimpleNew(2,dimensions,NPY_DOUBLE);

   /*
    * Copy
    *   I should write a specfile function that copies all data in a
    *   single pointer array
    */
    for (i=0;i<dimensions[0];i++) {
       for (j=0;j<dimensions[1];j++) {
          didx = j + i * dimensions[1];
          ((double *)PyArray_DATA(r_array))[didx] = data[j][i];
       }
    }
    /* memcpy(array->data,data,PyArray_NBYTES(array)); */

    freeArrNZ((void ***)&data,data_info[ROW]);
    free(data_info);
    if (data != (double **) NULL) {
        free(data);
    }
/*    return (PyObject *)array; */
    return PyArray_Return(r_array);
}

static PyObject *
scandata_dataline(PyObject *self,PyObject *args) {
    int     error;
    int     lineno;
    npy_intp ret;
    double  *data;

    PyArrayObject *r_array;

    SpecFile *sf;
    int     idx;

    scandataobject *s = (scandataobject *) self;

    sf  = (s->file)->sf;
    idx = s->index;

    if (!PyArg_ParseTuple(args,"i",&lineno))
            onError("cannot decode arguments for line data");

    ret = SfDataLine(sf,idx,lineno,&data,&error);

    if (ret == -1 )
          onError("cannot get data for line");

    r_array  = (PyArrayObject *)PyArray_SimpleNew(1,&ret,NPY_DOUBLE);

    memcpy(PyArray_DATA(r_array),data,PyArray_NBYTES(r_array));

    return (PyObject *)r_array;
}

static PyObject *
scandata_datacol(PyObject *self,PyObject *args) {
    int     error;
    int     colno;
    npy_intp ret;
    char    *colname;
    double  *data;

    PyArrayObject *r_array;

    SpecFile *sf;
    int     idx;
    scandataobject *s = (scandataobject *) self;

    sf  = (s->file)->sf;
    idx = s->index;

    if (!PyArg_ParseTuple(args,"i",&colno)) {
      PyErr_Clear() ;
      if (!PyArg_ParseTuple(args,"s",&colname)) {
    onError("cannot decode arguments for column data");
      } else {
    ret = SfDataColByName(sf,idx,colname,&data,&error);
      }
    } else {
      ret = SfDataCol(sf,idx,colno,&data,&error);
    }

    if (ret == -1 )
      onError("cannot get data for column");

    r_array      = (PyArrayObject *)PyArray_SimpleNew(1,&ret,NPY_DOUBLE);

   if (data != (double *) NULL){
        memcpy(PyArray_DATA(r_array),data,PyArray_NBYTES(r_array));
        free(data);
    }else{
        /* return an empty array? */
        printf("I should return an empty array ...\n");
        PyArray_FILLWBYTE(r_array, 0);
    }

    return PyArray_Return(r_array);
    /*
      it does not work for solaris and linux
      I should check the call to PyErr_Occurred()) in Numeric/Src/arrayobject.c
      PyArray_Return(array);
     */
}


static PyObject *
scandata_alllabels  (PyObject *self,PyObject *args)
{
    int error,i;
    char **labels;
    long nb_labels;
    PyObject *t,*x;

    scandataobject *v = (scandataobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    nb_labels = SfAllLabels((v->file)->sf,v->index,&labels,&error);

    t = PyList_New(nb_labels);
    for ( i = 0 ;i<nb_labels;i++) {
       x = PyUnicode_FromString(labels[i]);
       PyList_SetItem(t,i,x);
    }

    return t;
}

static PyObject *
scandata_allmotors  (PyObject *self,PyObject *args)
{
    int error,i;
    char **motors;
    long nb_motors;
    PyObject *t,*x;

    scandataobject *v = (scandataobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    nb_motors = SfAllMotors((v->file)->sf,v->index,&motors,&error);

    t = PyList_New(nb_motors);
    for ( i = 0 ;i<nb_motors;i++) {
       x = PyUnicode_FromString(motors[i]);
       PyList_SetItem(t,i,x);
    }

    return t;
}

static PyObject   *
scandata_allmotorpos  (PyObject *self,PyObject *args)
{
    int error,i;
    double *motorpos;
    long nb_motors;
    PyObject *t,*x;

    scandataobject *v = (scandataobject *) self;

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    nb_motors = SfAllMotorPos((v->file)->sf,v->index,&motorpos,&error);

    t = PyList_New(nb_motors);

    for ( i = 0 ;i<nb_motors;i++) {
       x = PyFloat_FromDouble(motorpos[i]);
       PyList_SetItem(t,i,x);
    }

    return t;
}

static PyObject   *
scandata_motorpos  (PyObject *self,PyObject *args)
{
    char   *motorname;
    int     error;
    double  motorpos;

    scandataobject *v = (scandataobject *) self;

    if (!PyArg_ParseTuple(args,"s",&motorname)) {
       return NULL;
    }

    motorpos = SfMotorPosByName((v->file)->sf,v->index,motorname,&error);

    if (motorpos != HUGE_VAL) {
        return Py_BuildValue("f",motorpos);
    } else {
        onError("cannot get position for motor");
    }

}

static PyObject *
scandata_hkl          (PyObject *self,PyObject *args)
{
    int idx,error;
    double *hkl;
    PyObject *pyo;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 ) {
        onError("empty scan data");
    }
    sf  = (s->file)->sf;

    hkl = SfHKL(sf,idx,&error);

    if (hkl == NULL)
        onError("cannot get data for column");

    pyo = Py_BuildValue("ddd",hkl[0],hkl[1],hkl[2]);
    free(hkl);
    return pyo;

}

static PyObject *
scandata_number       (PyObject *self,PyObject *args)
{
    int number,idx;
    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    sf  = (s->file)->sf;
    idx = s->index;

    number = SfNumber(sf,idx);

    return Py_BuildValue("i",number);

}

static PyObject   *
scandata_order        (PyObject *self,PyObject *args)
{
    int order,idx;
    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    sf  = (s->file)->sf;
    idx = s->index;

    order = SfOrder(sf,idx);

    return Py_BuildValue("i",order);

}

static PyObject   *
scandata_command      (PyObject *self,PyObject *args)
{
    int idx,error;
    char *command;
    PyObject *pyo;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 ) {
        onError("empty scan data");
    }
    sf  = (s->file)->sf;

    command = SfCommand(sf,idx,&error);

    if (command == NULL)
       onError("cannot get command for scan")

    pyo = Py_BuildValue("s",command);
    free(command);
    return pyo;

}

static PyObject   *
scandata_date      (PyObject *self,PyObject *args)
{
    int idx,error;
    char *date;
    PyObject *pyo;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 ) {
        onError("empty scan data");
    }
    sf  = (s->file)->sf;

    date = SfDate(sf,idx,&error);

    if (date == NULL)
        onError("cannot get date for scan");

    pyo =  Py_BuildValue("s",date);
    free(date);
    return pyo;

}

static PyObject   *
scandata_cols      (PyObject *self,PyObject *args)
{
    int cols,idx;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 )
        onError("empty scan data");

    cols = s->cols;

    if (cols == -1)
        onError("cannot get cols for scan");

    return Py_BuildValue("l",cols);
}

static PyObject   *
scandata_lines      (PyObject *self,PyObject *args)
{
    int lines,idx,error;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 )
        onError("empty scan data");

    sf  = (s->file)->sf;

    lines = SfNoDataLines(sf,idx,&error);

    /*if (lines == -1){*/
    if (lines < 0){
        onError("cannot get lines for scan");
        lines=0;
    }

    return Py_BuildValue("l",lines);
}

static PyObject   *
scandata_fileheader     (PyObject *self,PyObject *args)
{
    int i,no_lines,idx,error;
    char **lines;
    char *searchstr;
    PyObject *t,*x;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    if (!PyArg_ParseTuple(args,"s",&searchstr))
      return NULL;

    idx = s->index;
    if (idx == -1 )
        onError("empty scan data");

    sf  = (s->file)->sf;

    no_lines = SfFileHeader(sf,idx,searchstr,&lines,&error);

    if (no_lines == -1)
        onError("cannot get lines for scan");

    t = PyList_New(no_lines);
    for ( i = 0 ;i<no_lines;i++) {
       x = PyUnicode_FromString(lines[i]);
       PyList_SetItem(t,i,x);
    }

    return t;

    return Py_BuildValue("l",no_lines);
}

static PyObject   *
scandata_header     (PyObject *self,PyObject *args)
{
    int i,no_lines,idx,error;
    char **lines;
    char *searchstr;
    PyObject *t,*x;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    if (!PyArg_ParseTuple(args,"s",&searchstr))
      return NULL;

    idx = s->index;
    if (idx == -1 )
        onError("empty scan data");

    sf  = (s->file)->sf;

    no_lines = SfHeader(sf,idx,searchstr,&lines,&error);

    if (no_lines == -1)
        onError("cannot get lines for scan");

    t = PyList_New(no_lines);
    for ( i = 0 ;i<no_lines;i++) {
       x = PyUnicode_FromString(lines[i]);
       PyList_SetItem(t,i,x);
    }

    return t;

    return Py_BuildValue("l",no_lines);
}

static PyObject   *
scandata_nbmca      (PyObject *self,PyObject *args)
{
    int nomca,idx,error;

    PyObject *pyo;
    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 ) {
        onError("empty scan data");
    }
    sf  = (s->file)->sf;

    nomca = SfNoMca(sf,idx,&error);

    if (nomca == -1)
        onError("cannot get number of mca for scan");

    pyo =  Py_BuildValue("l",nomca);
    return pyo;
}

static PyObject   *
scandata_mca      (PyObject *self,PyObject *args)
{
    int    error;
    npy_intp ret;
    long   idx,mcano;

    double         *mcadata = NULL;
    PyArrayObject  *r_array;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    if (!PyArg_ParseTuple(args,"l",&mcano))
            onError("cannot decode arguments for line data");

    idx = s->index;

    if (idx == -1 ) {
        onError("empty scan data");
    }

    sf  = (s->file)->sf;

    ret = SfGetMca(sf,idx,mcano,&mcadata,&error);

    if (ret == -1)
        onError("cannot get mca for scan");

    r_array = (PyArrayObject *)PyArray_SimpleNew(1,&ret,NPY_DOUBLE);


    if (mcadata != (double *) NULL){
        memcpy(PyArray_DATA(r_array),mcadata,PyArray_NBYTES(r_array));
        free(mcadata);
    }else{
        printf("I should give back an empty array\n");
    }

/*    return (PyObject *)array; */

    return PyArray_Return(r_array);
    /*
      it does not work for solaris and linux
      I should check the call to PyErr_Occurred()) in Numeric/Src/arrayobject.c
      PyArray_Return(array);
     */
}

static PyObject   *
scandata_show      (PyObject *self,PyObject *args)
{
    int idx;

    SpecFile *sf;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;

    if (idx == -1 )
        onError("empty scan data");

    sf  = (s->file)->sf;

    SfShowScan(sf,idx);

    return Py_BuildValue("l",0);
}


   /*
    * Scandata basic python operations
    */

static PySequenceMethods scandata_as_sequence = {
  scandata_size,    /*     length len(sf)     */
  0,                /*     concat sf1 + sf2   */
  0,                /*     repeat sf * n      */
  scandata_col,     /*     item  sf[i], in    */
  scandata_slice,   /*     slice sf[i:j]      */
  0,                /*     asset sf[i] = v    */
  0,                /* slice ass. sf[i:j] = v */
};

static PyTypeObject Scandatatype = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "scandata",                /* tp_name */
    sizeof(scandataobject),    /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)scandata_free, /* tp_dealloc */
    (printfunc)scandata_print, /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    &scandata_as_sequence,     /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Scandata objects",        /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    scandata_methods,          /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};

static PyObject *
specfile_select(PyObject *self, PyObject *args)
{

    int  n,number,order,index;
    char   *scanstr;
    int error;
    scandataobject *v;
    specfileobject *f = (specfileobject *)self;

   if (!PyArg_ParseTuple(args,"s",&scanstr)) {
       return NULL;
   } else {
       n = sscanf(scanstr,"%d.%d",&number,&order);
       if ( n < 1 || n > 2 ) onError("cannot decode scan number/order");
       if ( n == 1) order = 1;
   }

    index = SfIndex(f->sf,number,order);

    if (index == -1 )
          onError("scan not found");

    Scandatatype.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Scandatatype) < 0)
        return NULL;

    v = (scandataobject *) _PyObject_New(&Scandatatype);
    if (v == NULL)
        return NULL;

    v->file  = f;
    v->index = index;
    v->cols  = SfNoColumns(f->sf, v->index,&error);
    /* increment reference to Specfile instance */
    Py_INCREF(self);
    return (PyObject *) v;
}

static PyObject *
specfile_scan(PyObject *self, Py_ssize_t index) {
    int error;

    scandataobject *v;
    specfileobject *f;
    f = (specfileobject *) self;

    if ( index < 0 || index >= f->length) {
         PyErr_SetString(PyExc_IndexError,"scan out of bounds");
         return NULL;
    }

    Scandatatype.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Scandatatype) < 0)
        return NULL;

    v = (scandataobject *) _PyObject_New(&Scandatatype);
    if (v == NULL)
        return NULL;

    v->file  = f;
    v->index = (int) index+1;
    v->cols  = SfNoColumns(f->sf,v->index,&error);

    Py_INCREF(self);

    return (PyObject *) v;
}

static PyObject *
scandata_free(PyObject *self) {
    scandataobject *s =(scandataobject *)self;
    specfileobject *f;
    f = s->file;
    Py_DECREF((PyObject *)f);
    PyObject_DEL(self);
    return NULL;
}

static int
scandata_print(PyObject *self,FILE *fp,int flags) {
    int ok=0;
    SpecFile *sf;
    int idx;

    scandataobject *s = (scandataobject *) self;

    idx = s->index;
    if (idx == -1 ) {
        fprintf(fp,"scandata('empty')");
    } else {
        sf  = (s->file)->sf;
        fprintf(fp,"scandata('source: %s,scan: %d.%d')",
                                    (s->file)->name,
                                    (int) SfNumber(sf,idx),
                                    (int) SfOrder (sf,idx));
    }
    return ok;
}

/*--------------------------------------------------*/
static PyMethodDef SpecfileMethods[] = {
    {"Specfile",  specfile_open, METH_VARARGS, "Open a Specfile instance."},
    {"system",  specfile_system, METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef specfilemodule = {
   PyModuleDef_HEAD_INIT,
   "specfile",   /* name of module */
   NULL, /*spam_doc,*/     /* module documentation, may be NULL */
   -1,             /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
   SpecfileMethods
};


PyMODINIT_FUNC
PyInit_specfile(void)
{
    PyObject *m;
    m = PyModule_Create(&specfilemodule);
    if (m == NULL)
        return NULL;
    import_array();
    SpecfileError = PyErr_NewException("specfile.error", NULL, NULL);
    Py_INCREF(SpecfileError);
    PyModule_AddObject(m, "error", SpecfileError);
    return m;
}

/*
 * Utility functions
 */
static char *
compList(long *nolist,long howmany)
{
     long this,colon;
     char buf[30];
     static char str[50000];
     char *retstr;

     if (howmany < 1) { return((char *)NULL);}

     sprintf(buf,"%d",(int) nolist[0]);

     *str = '\0';

     strcat(str,buf);

     colon=0;
     for(this=1;this<howmany;this++) {
         if ((nolist[this] - nolist[this-1]) == 1) {
            colon = 1;
         } else {
            if (colon) {
               sprintf(buf,":%d,%d",(int) nolist[this-1],(int) nolist[this]);
               colon=0;
            } else {
               sprintf(buf,",%d",(int) nolist[this]);
            }
            strcat(str,buf);
         }
     }

     if (howmany != 1 ) {
        if (colon) {
           sprintf(buf,":%d",(int) nolist[howmany-1]);
           strcat(str,buf);
        }
     }
     retstr = (char *)strdup(str);
     return(retstr);
}
