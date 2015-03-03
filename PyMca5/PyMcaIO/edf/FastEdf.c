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
/* FastEdfo objects */
#include <Python.h>
#include <./numpy/arrayobject.h>
#include <stdio.h>


static PyObject *ErrorObject;

typedef struct {
	PyObject_HEAD
	PyObject	*x_attr;	/* Attributes dictionary */
} FastEdfoObject;

staticforward PyTypeObject FastEdfo_Type;

/*
 * Function prototypes
 */
static FastEdfoObject *newFastEdfoObject (PyObject *arg);
static void                FastEdfo_dealloc  (FastEdfoObject *self);
static int Alen = 0;

#define FastEdfoObject_Check(v)	((v)->ob_type == &FastEdfo_Type)

static FastEdfoObject *
newFastEdfoObject(arg)
	PyObject *arg;
{
	FastEdfoObject *self;
	self = PyObject_NEW(FastEdfoObject, &FastEdfo_Type);
	if (self == NULL)
		return NULL;
	self->x_attr = NULL;
	return self;
}

/* FastEdfo methods */

static void
FastEdfo_dealloc(self)
	FastEdfoObject *self;
{
	Py_XDECREF(self->x_attr);
	PyObject_DEL(self);
}

/*
static PyObject *
FastEdfo_demo(self, args)
	FastEdfoObject *self;
	PyObject *args;
*/
static PyObject *
FastEdfo_demo(FastEdfoObject *self,
	PyObject *args)
{
	if (!PyArg_ParseTuple(args, ":demo"))
		return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef FastEdfo_methods[] = {
	{"demo",	(PyCFunction)FastEdfo_demo,	1},
	{NULL,		NULL}		/* sentinel */
};

static PyObject *
FastEdfo_getattr(FastEdfoObject *self,
	char *name)
{
	if (self->x_attr != NULL) {
		PyObject *v = PyDict_GetItemString(self->x_attr, name);
		if (v != NULL) {
			Py_INCREF(v);
			return v;
		}
	}
	return Py_FindMethod(FastEdfo_methods, (PyObject *)self, name);
}

static int
FastEdfo_setattr(FastEdfoObject *self, char *name,
	PyObject *v)
{
	if (self->x_attr == NULL) {
		self->x_attr = PyDict_New();
		if (self->x_attr == NULL)
			return -1;
	}
	if (v == NULL) {
		int rv = PyDict_DelItemString(self->x_attr, name);
		if (rv < 0)
			PyErr_SetString(PyExc_AttributeError,
			        "delete non-existing FastEdfo attribute");
		return rv;
	}
	else
		return PyDict_SetItemString(self->x_attr, name, v);
}

statichere PyTypeObject FastEdfo_Type = {
	/* The ob_type field must be initialized in the module init function
	 * to be portable to Windows without using C++. */
	PyObject_HEAD_INIT(NULL)
	0,			/*ob_size*/
	"FastEdfo",			/*tp_name*/
	sizeof(FastEdfoObject),	/*tp_basicsize*/
	0,			/*tp_itemsize*/
	/* methods */
	(destructor)FastEdfo_dealloc, /*tp_dealloc*/
	0,			/*tp_print*/
	(getattrfunc)FastEdfo_getattr, /*tp_getattr*/
	(setattrfunc)FastEdfo_setattr, /*tp_setattr*/
	0,			/*tp_compare*/
	0,			/*tp_repr*/
	0,			/*tp_as_number*/
	0,			/*tp_as_sequence*/
	0,			/*tp_as_mapping*/
	0,			/*tp_hash*/
};
/* --------------------------------------------------------------------- */


static PyObject *FastEdf_extended_fread(PyObject *self, PyObject *args)
{
    PyObject *resultobj;
    char *arg1 ;
    int arg2 ;
    int arg3 ;
    int *arg4 = (int *) 0 ;
    int *arg5 = (int *) 0 ;
    FILE *arg6 = (FILE *) 0 ;
    PyArrayObject *tmp1 = NULL ;
    PyArrayObject *tmp3 = NULL ;
    PyArrayObject *tmp4 = NULL ;
    PyObject * obj0  = 0 ;
    PyObject * obj2  = 0 ;
    PyObject * obj3  = 0 ;
    PyObject * obj4  = 0 ;
    void extended_fread(char *, int, int,int *,int *,FILE *);
            long totalsize = 1;
            int sizeofunit=0;
            int i;

    if(!PyArg_ParseTuple(args,(char *)"OiOOO:extended_fread",&obj0,&arg2,&obj2,&obj3,&obj4)) goto fail;
    {
        tmp1 = (PyArrayObject *) obj0;
        if((tmp1)->flags %2 == 0)  {
            PyErr_SetString(PyExc_ValueError," array has to be contiguous" );
            return NULL;
        }
        arg1 = (char  *)tmp1->data;
    }
    {
        tmp3 = (PyArrayObject *)PyArray_ContiguousFromObject(obj2, PyArray_INT, 1, 1);
        if(tmp3 == NULL) return NULL;
        arg3 = Alen = tmp3->dimensions[0];
        arg4 = (int *)tmp3->data;
    }
    {
        tmp4 = (PyArrayObject *)PyArray_ContiguousFromObject(obj3, PyArray_INT, 1, 1);
        if(tmp4 == NULL) return NULL;
        if(tmp4->dimensions[0] != Alen) {
            PyErr_SetString(PyExc_ValueError, "Vectors must be same length.");
            return NULL;
        }
        arg5 = (int *)tmp4->data;

        {
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_CHAR ) sizeofunit=1;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_UBYTE ) sizeofunit=1;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_BYTE ) sizeofunit=1;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_SHORT ) sizeofunit=2;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_INT )   sizeofunit=4;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_LONG ) sizeofunit=4;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_FLOAT ) sizeofunit=4;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_DOUBLE ) sizeofunit=8;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_CFLOAT ) sizeofunit=8;
            if (  ((PyArrayObject *) obj0)->descr->type_num == PyArray_CDOUBLE ) sizeofunit=16;

            for(i=0; i<arg3; i++ ) {
                totalsize *= arg4[i] ;
            }
            if ( ( PyArray_Size( ( obj0) ))  != totalsize*arg2/sizeofunit ) {
                printf("needed size = %li\n",totalsize*arg2/sizeofunit);
                PyErr_SetString(PyExc_ValueError, "You provided an array of the wrong size");
                return NULL;
            }

        }

    }
    {
        arg6 = PyFile_AsFile(obj4);
    }
    extended_fread(arg1,arg2,arg3,arg4,arg5,arg6);

    Py_INCREF(Py_None); resultobj = Py_None;
    {
        if(tmp3){
            Py_DECREF(tmp3);
        }
    }
    {
        if(tmp4) {
            Py_DECREF(tmp4);
        }
    }
    return resultobj;
    fail:
    {
        if(tmp3) {
        Py_DECREF(tmp3);
        }
    }
    {
        if(tmp4) {
            Py_DECREF(tmp4);
        }
    }
    return NULL;
}


void extended_fread(   char *ptr,      /* memory to write in */
                       int   size_of_block,
                       int ndims        ,
                       int *dims      ,
                       int  *strides    ,
                       FILE *stream  ) {
  int pos;
  int oldpos;
  int  count;

#ifdef WIN32
  int indexes[100];
#else
  int indexes[ndims];
#endif
  int i;
  int loop;
  int res;

  oldpos=0;
  pos=0;
  count = 0;
  /*
  printf("received\n");
    printf("block = %d\n",size_of_block);
    printf("ndims = %d\n",ndims);
    printf("dims = %d %d\n",dims[0],dims[1]);
    printf("strides = %d\n",strides[0]);

*/
  for(i=0; i<ndims; i++) {
    indexes[i]=0;
  }
  loop=ndims-1;
  indexes[ndims-1 ]=-1  ;
  pos=-strides[ndims-1];

  while(1) {
    if(indexes[loop]< dims[loop]-1 ) {
      indexes[loop]++;
      pos += strides[loop];
      for( i=loop+1; i<ndims; i++)  {
        pos -= indexes[i]*strides[i];
        indexes[i]=0;
      }
      res=fseek( stream, (pos-oldpos),  SEEK_CUR );
      if(res!=0) {
    /*    throw ErrorExtendedFread_fseek_failed();*/
    printf("Error 1/n");
        break;
      }

      res=fread (  ( (char * ) ptr) +(count)*size_of_block ,
                   size_of_block, 1,  stream );
      if(res!=1) {
    /*    throw ErrorExtendedFread_fseek_failed();*/
    printf("Error 2/n");
        break;
      }
      count++;

      oldpos = pos+ size_of_block;

      loop=ndims-1;
      /*      for(i=0 ; i< ndims; i++) {
              printf(" %d ", indexes[i] );
              }
              printf("\n");
      */
    }else {
      loop--;
    }
    if(loop==-1) {
      break;
    }
  }
}



/* List of functions defined in the module */

static PyMethodDef FastEdf_methods[] = {
    {"extended_fread",      FastEdf_extended_fread,   METH_VARARGS},
	{NULL,		NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFastEdf) */

DL_EXPORT(void)
initFastEdf(void)
{
	PyObject *m, *d;

	/* Initialize the type of the new type object here; doing it here
	 * is required for portability to Windows without requiring C++. */
	FastEdfo_Type.ob_type = &PyType_Type;

	/* Create the module and add the functions */
	m = Py_InitModule("FastEdf", FastEdf_methods);
    import_array();

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyErr_NewException("FastEdf.error", NULL, NULL);
	PyDict_SetItemString(d, "error", ErrorObject);
}
