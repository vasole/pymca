#include <Python.h>
#include <./numpy/arrayobject.h>

/* static variables */

static PyObject *PyMcaIOHelperError;

/* Function declarations */

static PyObject *PyMcaIOHelper_fillSupaVisio(PyObject *dummy, PyObject *args);

/* ------------------------------------------------------- */
static PyObject *
PyMcaIOHelper_fillSupaVisio(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject *inputArray, *outputArray;
    int nChannels = 2048;
    unsigned short *dataPointer, x, y, ch;
    int i, dimensions[3];
    int maxx, maxy, maxch;
    unsigned int *outputPointer;

    if (!PyArg_ParseTuple(args, "O|d", &input, &nChannels))
        return NULL;
    inputArray = (PyArrayObject *)
             	PyArray_ContiguousFromObject(input, PyArray_USHORT, 2,2);
    if (inputArray == NULL)
    {
	PyErr_SetString(PyMcaIOHelperError, "Cannot parse input array");
        return NULL;
    }
	
    dataPointer = (unsigned short *) inputArray->data;
    dataPointer++;
    dimensions[1] = *dataPointer++;
    dimensions[0] = *dataPointer++;
    dimensions[2] = nChannels;
    outputArray = (PyArrayObject *)
	    PyArray_FromDims(3, dimensions, PyArray_UINT);

    /* Do the job */
    maxx=maxy=maxch=0;
    outputPointer = (unsigned int *) outputArray->data; 
    for (i=3; i<inputArray->dimensions[0]; i++)
    {
        y = *dataPointer++;
	x = *dataPointer++;
	ch = *dataPointer++;
	if (ch >= nChannels)
	{
		printf("bad reading %d\n", ch);
		continue;
	}
	*(outputPointer+ (dimensions[1] * x + y) * nChannels + ch) += 1;
    }
    Py_DECREF(inputArray);
    return PyArray_Return(outputArray);
}

/* Module methods */
static PyMethodDef PyMcaIOHelperMethods[] ={
	{"fillSupaVisio", PyMcaIOHelper_fillSupaVisio, METH_VARARGS},
	{NULL,NULL, 0, NULL} /* sentinel */
};

/* Initialise the module */
PyMODINIT_FUNC 
initPyMcaIOHelper(void)
{
	PyObject *m, *d;
	/* Create the module and add the functions */
	m = Py_InitModule("PyMcaIOHelper", PyMcaIOHelperMethods);

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);

	import_array();
	PyMcaIOHelperError = PyErr_NewException("PyMcaIOHelper.error", NULL, NULL);
	PyDict_SetItemString(d, "error", PyMcaIOHelperError);
}

