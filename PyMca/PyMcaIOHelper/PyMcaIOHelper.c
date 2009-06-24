#include <Python.h>
#include <./numpy/arrayobject.h>

/* static variables */

static PyObject *PyMcaIOHelperError;

/* Function declarations */

static PyObject *PyMcaIOHelper_fillSupaVisio(PyObject *dummy, PyObject *args);
static PyObject *PyMcaIOHelper_readAifira(PyObject *dummy, PyObject *args);

/* ------------------------------------------------------- */
static PyObject *
PyMcaIOHelper_fillSupaVisio(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject *inputArray, *outputArray;
    int nChannels = 2048;
    unsigned short *dataPointer, x, y, ch;
    int i;
    npy_intp dimensions[3];
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
    outputArray = (PyArrayObject *) PyArray_SimpleNew(3, dimensions, PyArray_UINT);
    PyArray_FILLWBYTE(outputArray, 0);
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

static PyObject *
PyMcaIOHelper_readAifira(PyObject *self, PyObject *args)
{
    PyObject *inputFileDescriptor;
    FILE *fd;
    PyArrayObject *outputArray;
    int nChannels = 2048;
    unsigned short channel;
    unsigned char x, y; 
    npy_intp dimensions[3];
    unsigned int *outputPointer;

    if (!PyArg_ParseTuple(args, "O", &inputFileDescriptor))
    {
        PyErr_SetString(PyMcaIOHelperError, "Error parsing input arguments");
        return NULL;
    }
    if (!PyFile_Check(inputFileDescriptor))
    {
        PyErr_SetString(PyMcaIOHelperError, "Input is not a python file descriptor object");
        return NULL;
    }
    fd = PyFile_AsFile(inputFileDescriptor);

    dimensions[0] = 128;
    dimensions[1] = 128;
    dimensions[2] = nChannels;

    outputArray = (PyArrayObject *) PyArray_SimpleNew(3, dimensions, PyArray_UINT);
    PyArray_FILLWBYTE(outputArray, 0);

    /* Do the job */
    outputPointer = (unsigned int *) outputArray->data;
    while(fscanf(fd, "%2c%c%c", (char *)&channel, &x, &y) == 3)
    {
        if (channel >= nChannels)
        {
            printf("bad reading %d\n", channel);
            continue;
        }
        if (x < 1)
            continue;
        if (y < 1)
            continue;
        if (x > 128)
        {
            printf("bad X reading %d\n", x);
            break;
            continue;
        }
        if (y > 128)
        {
            printf("bad Y reading %d\n", y);
            break;
            continue;
        }
        x -= 1;
        y -= 1;
        /* normally pixe data are in the second channel */
        if (channel > 1023)
        {
            channel -= 1024;
        }
        else
            channel += 1024;

        *(outputPointer + (dimensions[1] * x + y) * nChannels + channel) += 1;
    }
    return PyArray_Return(outputArray);
}

/* Module methods */
static PyMethodDef PyMcaIOHelperMethods[] ={
    {"fillSupaVisio", PyMcaIOHelper_fillSupaVisio, METH_VARARGS},
    {"readAifira", PyMcaIOHelper_readAifira, METH_VARARGS},
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

