/* Subset of SIGTOOLS module by Travis Oliphant

Copyright 2005 Travis Oliphant
Permission to use, copy, modify, and distribute this software without fee
is granted under the SciPy License.
*/

#include "Python.h"
#include "numpy/noprefix.h"

#include <setjmp.h>

typedef struct {
  char *data;
  int elsize;
} Generic_ptr;

typedef struct {
  char *data;
  intp numels;
  int elsize;
  char *zero;        /* Pointer to Representation of zero */
} Generic_Vector;

typedef struct {
  char *data;
  int  nd;
  intp  *dimensions;
  int  elsize;
  intp  *strides;
  char *zero;         /* Pointer to Representation of zero */
} Generic_Array;

#define PYERR(message) {PyErr_SetString(PyExc_ValueError, message); goto fail;}

#define DATA(arr) ((arr)->data)
#define DIMS(arr) ((arr)->dimensions)
#define STRIDES(arr) ((arr)->strides)
#define ELSIZE(arr) ((arr)->descr->elsize)
#define OBJECTTYPE(arr) ((arr)->descr->type_num)
#define BASEOBJ(arr) ((PyArrayObject *)((arr)->base))
#define RANK(arr) ((arr)->nd)
#define ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)


jmp_buf MALLOC_FAIL;

char *check_malloc (int);

char *check_malloc (int size)
{
    char *the_block;
    
    the_block = (char *)malloc(size);
    if (the_block == NULL)
	{
	    printf("\nERROR: unable to allocate %d bytes!\n", size);
	    longjmp(MALLOC_FAIL,-1);
	}
    return(the_block);
}

   
static char doc_median2d[] = "filt = _median2d(data, size)";

extern void f_medfilt2(float*,float*,intp*,intp*);
extern void d_medfilt2(double*,double*,intp*,intp*);
extern void b_medfilt2(unsigned char*,unsigned char*,intp*,intp*);
extern void short_medfilt2(short*, short*,intp*,intp*);
extern void ushort_medfilt2(unsigned short*,unsigned short*,intp*,intp*);
extern void int_medfilt2(int*, int*,intp*,intp*);
extern void uint_medfilt2(unsigned int*,unsigned int*,intp*,intp*);
extern void long_medfilt2(long*, long*,intp*,intp*);
extern void ulong_medfilt2(unsigned long*,unsigned long*,intp*,intp*);

static PyObject *mediantools_median2d(PyObject *dummy, PyObject *args)
{
    PyObject *image=NULL, *size=NULL;
    int typenum;
    PyArrayObject *a_image=NULL, *a_size=NULL;
    PyArrayObject *a_out=NULL;
    intp Nwin[2] = {3,3};

    if (!PyArg_ParseTuple(args, "O|O", &image, &size)) return NULL;

    typenum = PyArray_ObjectType(image, 0);
    a_image = (PyArrayObject *)PyArray_ContiguousFromObject(image, typenum, 2, 2);
    if (a_image == NULL) goto fail;

    if (size != NULL) {
	a_size = (PyArrayObject *)PyArray_ContiguousFromObject(size, PyArray_LONG, 1, 1);
	if (a_size == NULL) goto fail;
	if ((RANK(a_size) != 1) || (DIMS(a_size)[0] < 2)) 
	    PYERR("Size must be a length two sequence");
	Nwin[0] = ((intp *)DATA(a_size))[0];
	Nwin[1] = ((intp *)DATA(a_size))[1];
    }  

    a_out = (PyArrayObject *)PyArray_SimpleNew(2,DIMS(a_image),typenum);
    if (a_out == NULL) goto fail;

    if (setjmp(MALLOC_FAIL)) {
	PYERR("Memory allocation error.");
    }
    else {
	switch (typenum) {
	case PyArray_UBYTE:
	    b_medfilt2((unsigned char *)DATA(a_image), (unsigned char *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_FLOAT:
	    f_medfilt2((float *)DATA(a_image), (float *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_DOUBLE:
	    d_medfilt2((double *)DATA(a_image), (double *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_SHORT:
	    short_medfilt2((short *)DATA(a_image), (short *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_USHORT:
	    ushort_medfilt2((unsigned short *)DATA(a_image), (unsigned short *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_INT:
	    int_medfilt2((int *)DATA(a_image), (int *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_UINT:
	    uint_medfilt2((unsigned int *)DATA(a_image), (unsigned int *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_LONG:
	    long_medfilt2((long *)DATA(a_image), (long *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	case PyArray_ULONG:
	    ulong_medfilt2((unsigned long *)DATA(a_image), (unsigned long *)DATA(a_out), Nwin, DIMS(a_image));
	    break;
	default:
	  PYERR("2D median filter only supports Int8, Float32, and Float64.");
	}
    }

    Py_DECREF(a_image);
    Py_XDECREF(a_size);

    return PyArray_Return(a_out);
 
 fail:
    Py_XDECREF(a_image);
    Py_XDECREF(a_size);
    Py_XDECREF(a_out);
    return NULL;

}

static struct PyMethodDef toolbox_module_methods[] = {
	{"_medfilt2d", mediantools_median2d, METH_VARARGS, doc_median2d},
	{NULL,		NULL, 0}		/* sentinel */
};

/* Initialization function for the module (*must* be called initmediantools) */

PyMODINIT_FUNC initmediantools(void) {
        PyObject *m, *d;
	
	/* Create the module and add the functions */
	m = Py_InitModule("mediantools", toolbox_module_methods);

	/* Import the C API function pointers for the Array Object*/
	import_array();

	/* Make sure the multiarraymodule is loaded so that the zero
	   and one objects are defined */
	/* XXX: This should be updated for scipy. I think it's pulling in 
	   Numeric's multiarray. */
	PyImport_ImportModule("numpy.core.multiarray");
	/* { PyObject *multi = PyImport_ImportModule("multiarray"); } */

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);

	/* Check for errors */
	if (PyErr_Occurred()) {
	  PyErr_Print();
	  Py_FatalError("can't initialize module array");
	}
}
