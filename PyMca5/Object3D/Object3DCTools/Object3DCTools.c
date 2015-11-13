#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
#ifdef WIN32
#include <windows.h>
#endif
#include <Python.h>
#include <./numpy/arrayobject.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glext.h> /*GL_MAX_ELEMENTS_VERTICES is there*/
#else
#include <GL/gl.h>
#include <GL/glext.h> /*GL_MAX_ELEMENTS_VERTICES is there*/
#endif


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif


/* Function declarations */
static PyObject *getGridFacetsFromVertices(PyObject *dummy, PyObject *args);
static PyObject *testOpenGL(PyObject *dummy, PyObject *args);
static PyObject *gridMarchingCubes(PyObject *dummy, PyObject *args);
static PyObject *marchingCubesXYZ(PyObject *dummy, PyObject *args);
static PyObject *get2DGridFromXY(PyObject *dummy, PyObject *args);
static PyObject *draw2DGridPoints(PyObject *dummy, PyObject *args);
static PyObject *draw2DGridLines(PyObject *dummy, PyObject *args);
static PyObject *draw2DGridQuads(PyObject *dummy, PyObject *args);
static PyObject *get3DGridFromXYZ(PyObject *dummy, PyObject *args);
static PyObject *draw3DGridPoints(PyObject *dummy, PyObject *args);
static PyObject *draw3DGridLines(PyObject *dummy, PyObject *args);
static PyObject *draw3DGridQuads(PyObject *dummy, PyObject *args);
static PyObject *drawXYZPoints(PyObject *dummy, PyObject *args);
static PyObject *drawXYZLines(PyObject *dummy, PyObject *args);
static PyObject *drawXYZTriangles(PyObject *dummy, PyObject *args);
static PyObject *getVertexArrayMeshAxes(PyObject *dummy, PyObject *args);
static PyObject *draw3DGridTexture(PyObject *dummy, PyObject *args);
static int check2DGridVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xArray,\
						 PyArrayObject **yArray, PyArrayObject **zArray, PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xSize, npy_intp *ySize, npy_intp *zSize, npy_intp *cSize, npy_intp *vSize);

static int check3DGridVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xArray,\
						 PyArrayObject **yArray, PyArrayObject **zArray, PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xSize, npy_intp *ySize, npy_intp *zSize, npy_intp *cSize, npy_intp *vSize);

static int checkXYZVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xyzArray,\
						 PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 PyArrayObject **facetsArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xyzSize, npy_intp *cSize, npy_intp *vSize, npy_intp *facetsSize);

static PyObject *get2DGridFromXY(PyObject *self, PyObject *args)
{
	/* One can do this in pure python, but our goal is to keep memory
	   needs as low as possible in the process:
        #fast method to generate the vertices
        self.vertices = numpy.zeros((xsize * ysize, 3), numpy.float32)
        A=numpy.outer(x, numpy.ones(len(y), numpy.float32))
        B=numpy.outer(y, numpy.ones(len(x), numpy.float32))

        #the width is the number of columns
        self.vertices[:,0]=A.flatten()
        self.vertices[:,1]=B.transpose().flatten()
        self.vertices[:,2]=data.flatten()

	*/

	/* input parameters */
	PyObject	   *xinput, *yinput;
	PyArrayObject	*xArray, *yArray, *ret;
	npy_intp	    xSize, ySize;
	npy_intp	    i, j;
	npy_intp		dim[2];
	float			*px, *py, *pr;
    struct module_state *st = GETSTATE(self);

	if (!PyArg_ParseTuple(args, "OO", &xinput, &yinput))
	{
		PyErr_SetString(st->error, "Unable to parse arguments. Two float arrays required");
        return NULL;
	}

	/* convert to a contiguous array of at least 1 dimension */
	xArray = (PyArrayObject *)
    				PyArray_FROMANY(xinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (xArray == NULL)
	{
		PyErr_SetString(st->error, "First argument cannot be converted to a float array.");
        return 0;
	}

	yArray = (PyArrayObject *)
    				PyArray_FROMANY(yinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (yArray == NULL)
	{
		Py_DECREF(xArray);
		PyErr_SetString(st->error, "Second argument cannot be converted to a float array.");
        return 0;
	}

	/* obtain the size of the arrays */
	xSize = 1;
	for (i=0; i< (xArray)->nd;i++){
		xSize *= (xArray)->dimensions[i];
	}

	ySize = 1;
	for (i=0; i< (yArray)->nd;i++){
		ySize *= (yArray)->dimensions[i];
	}


	/* create the output array */
	dim[0] = xSize * ySize;
	dim[1] = 2;
    ret = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT);
    if (ret == NULL){
        Py_DECREF(xArray);
        Py_DECREF(yArray);
	    PyErr_SetString(st->error, "Error creating output array");
		return NULL;
    }

	/* fill the output array */
	pr = (float *) ret->data;
	px = (float *) xArray->data;
	for (i=0; i<xSize; i++){
		py = (float *) yArray->data;
		for (j=0; j<ySize; j++){
			*pr = *px;
			pr++;
			*pr = *py;
			pr++;
			py++;
		}
		px++;
	}

    Py_DECREF(xArray);
    Py_DECREF(yArray);
    return PyArray_Return(ret);
}


static PyObject *get3DGridFromXYZ(PyObject *self, PyObject *args)
{
	/* input parameters */
	PyObject	   *xinput, *yinput, *zinput;
	PyArrayObject	*xArray, *yArray, *zArray, *ret;
	npy_intp	    xSize, ySize, zSize;
	npy_intp	    i, j, k;
	npy_intp		dim[2];
	float			*px, *py, *pz, *pr;
    struct module_state *st = GETSTATE(self);

	if (!PyArg_ParseTuple(args, "OOO", &xinput, &yinput, &zinput))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. Three float arrays required");
        return NULL;
	}

	/* convert to a contiguous array of at least 1 dimension */
	xArray = (PyArrayObject *)
    				PyArray_FROMANY(xinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (xArray == NULL)
	{
	    PyErr_SetString(st->error, "First argument cannot be converted to a float array.");
        return 0;
	}

	yArray = (PyArrayObject *)
    				PyArray_FROMANY(yinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (yArray == NULL)
	{
		Py_DECREF(xArray);
		PyErr_SetString(st->error, "Second argument cannot be converted to a float array.");
        return 0;
	}

	zArray = (PyArrayObject *)
    				PyArray_FROMANY(zinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (zArray == NULL)
	{
		Py_DECREF(xArray);
		Py_DECREF(yArray);
	    PyErr_SetString(st->error, "Third argument cannot be converted to a float array.");
        return 0;
	}

	/* obtain the size of the arrays */
	xSize = 1;
	for (i=0; i< (xArray)->nd;i++){
		xSize *= (xArray)->dimensions[i];
	}

	ySize = 1;
	for (i=0; i< (yArray)->nd;i++){
		ySize *= (yArray)->dimensions[i];
	}

	zSize = 1;
	for (i=0; i< (zArray)->nd;i++){
		zSize *= (zArray)->dimensions[i];
	}

	/* create the output array */
	dim[0] = xSize * ySize * zSize;
	dim[1] = 3;
    ret = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_FLOAT);
    if (ret == NULL){
        Py_DECREF(xArray);
        Py_DECREF(yArray);
        Py_DECREF(zArray);
	    PyErr_SetString(st->error, "Error creating output array");
		return NULL;
    }

	/* fill the output array */
	pr = (float *) ret->data;
	px = (float *) xArray->data;
	for (i=0; i<xSize; i++){
		py = (float *) yArray->data;
		for (j=0; j<ySize; j++){
			pz = (float *) zArray->data;
			for (k=0; k<zSize; k++){
				*pr = *px;
				pr++;
				*pr = *py;
				pr++;
				*pr = *pz;
				pr++;
				pz++;
			}
			py++;
		}
		px++;
	}

    Py_DECREF(xArray);
    Py_DECREF(yArray);
    Py_DECREF(zArray);
    return PyArray_Return(ret);
}

static int check2DGridVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xArray,\
						 PyArrayObject **yArray, PyArrayObject **zArray, PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xSize, npy_intp *ySize, npy_intp *zSize, npy_intp *cSize, npy_intp *vSize)
{
	/* input parameters */
	PyObject	   *xinput, *yinput, *zinput, *cinput=NULL, *vinput=NULL;
	int			   cfilter=0, vfilter=0;
	float		   vmin=1, vmax=0;

	/* local variables */
	npy_intp	    i;
	struct module_state *st = GETSTATE(self);

	/* statements */
	if (!PyArg_ParseTuple(args, "OOO|OOi(iff)", &xinput, &yinput, &zinput, &cinput, &vinput, &cfilter, &vfilter, &vmin, &vmax))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. At least three float arrays required");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*xArray = (PyArrayObject *)
    				PyArray_FROMANY(xinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (*xArray == NULL)
	{
	    PyErr_SetString(st->error, "First argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*yArray = (PyArrayObject *)
    				PyArray_FROMANY(yinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (*yArray == NULL)
	{
		Py_DECREF(*xArray);
	    PyErr_SetString(st->error, "Second argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*zArray = (PyArrayObject *)
    				PyArray_FROMANY(zinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);

    if (*zArray == NULL)
	{
		Py_DECREF(*xArray);
		Py_DECREF(*yArray);
	    PyErr_SetString(st->error, "Third argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	if ((cinput != NULL) && (cinput != Py_None)){
		*colorArray = (PyArrayObject *)
    					PyArray_ContiguousFromAny(cinput, NPY_UBYTE, 1, 0);
		if (*colorArray == NULL)
		{
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			PyErr_SetString(st->error, "Fourth argument cannot be converted to an unsigned byte array.");
			return 0;
		}
	}

	/* obtain the size of the arrays */
	*xSize = 1;
	for (i=0; i< (*xArray)->nd;i++){
		*xSize *= (*xArray)->dimensions[i];
	}

	*ySize = 1;
	for (i=0; i< (*yArray)->nd;i++){
		*ySize *= (*yArray)->dimensions[i];
	}

	*zSize = 1;
	for (i=0; i< (*zArray)->nd;i++){
		*zSize *= (*zArray)->dimensions[i];
	}

	if (*zSize != (*xSize) * (*ySize)){
		PyErr_SetString(st->error, "Number of Z values does not match number of vertices.");
		return 0;
	}

	if ((cinput != NULL) && (cinput != Py_None)){
		*cSize = 1;
		for (i=0; i< (*colorArray)->nd;i++){
			*cSize *= (*colorArray)->dimensions[i];
		}

		if (*cSize != (4 * (*zSize))){
				Py_DECREF(*xArray);
				Py_DECREF(*yArray);
				Py_DECREF(*zArray);
				Py_DECREF(*colorArray);
				PyErr_SetString(st->error, "Number of colors does not match number of vertices.");
				return 0;
		}
	}

	/* convert to a contiguous array of at least 1 dimension */
	if ((vinput != NULL) && (vinput != Py_None)){
		*valuesArray = (PyArrayObject *)
    				PyArray_FROMANY(vinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
		if (*valuesArray == NULL)
		{
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			if (cinput != NULL)
			{
				Py_DECREF(*colorArray);
			}
			PyErr_SetString(st->error, "Values array cannot be converted to a float array.");
			return 0;
		}
		/* check the values array size */
		*vSize = 1;
		for (i=0; i< (*valuesArray)->nd;i++){
			*vSize *= (*valuesArray)->dimensions[i];
		}

		if (*vSize != *zSize){
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			if (cinput != NULL)
			{
				Py_DECREF(*colorArray);
			}
			Py_DECREF(*valuesArray);
			PyErr_SetString(st->error, "Number of values does not match number of vertices.");
			return 0;
		}
	}

	*colorFilterFlag = cfilter;
	*valueFilterFlag = vfilter;
	*vMin = vmin;
	*vMax = vmax;
	return 1;
}


static int check3DGridVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xArray,\
						 PyArrayObject **yArray, PyArrayObject **zArray, PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xSize, npy_intp *ySize, npy_intp *zSize, npy_intp *cSize, npy_intp *vSize)
{
	/* input parameters */
	PyObject	   *xinput, *yinput, *zinput, *cinput=NULL, *vinput=NULL;
	int			   cfilter=0, vfilter=0;
	float		   vmin=1, vmax=0;

	/* local variables */
	npy_intp	    i;
	struct module_state *st = GETSTATE(self);

	/* statements */
	if (!PyArg_ParseTuple(args, "OOO|OOi(iff)", &xinput, &yinput, &zinput, &cinput, &vinput, &cfilter, &vfilter, &vmin, &vmax))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. At least three float arrays required");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*xArray = (PyArrayObject *)
    				PyArray_FROMANY(xinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (*xArray == NULL)
	{
	    PyErr_SetString(st->error, "First argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*yArray = (PyArrayObject *)
    				PyArray_FROMANY(yinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (*yArray == NULL)
	{
		Py_DECREF(*xArray);
	    PyErr_SetString(st->error, "Second argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*zArray = (PyArrayObject *)
    				PyArray_FROMANY(zinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);

    if (*zArray == NULL)
	{
		Py_DECREF(*xArray);
		Py_DECREF(*yArray);
	    PyErr_SetString(st->error, "Third argument cannot be converted to a float array.");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	if ((cinput != NULL) && (cinput != Py_None)){
		*colorArray = (PyArrayObject *)
    					PyArray_ContiguousFromAny(cinput, NPY_UBYTE, 1, 0);
		if (*colorArray == NULL)
		{
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			PyErr_SetString(st->error, "Fourth argument cannot be converted to an unsigned byte array.");
			return 0;
		}
	}

	/* obtain the size of the arrays */
	*xSize = 1;
	for (i=0; i< (*xArray)->nd;i++){
		*xSize *= (*xArray)->dimensions[i];
	}

	*ySize = 1;
	for (i=0; i< (*yArray)->nd;i++){
		*ySize *= (*yArray)->dimensions[i];
	}

	*zSize = 1;
	for (i=0; i< (*zArray)->nd;i++){
		*zSize *= (*zArray)->dimensions[i];
	}

	if ((cinput != NULL) && (cinput != Py_None)){
		*cSize = 1;
		for (i=0; i< (*colorArray)->nd;i++){
			*cSize *= (*colorArray)->dimensions[i];
		}

		if (*cSize != (4 * (*xSize) * (*ySize) *  (*zSize))){
				Py_DECREF(*xArray);
				Py_DECREF(*yArray);
				Py_DECREF(*zArray);
				Py_DECREF(*colorArray);
				PyErr_SetString(st->error, "Number of colors does not match number of vertices.");
				return 0;
		}
	}

	/* convert to a contiguous array of at least 1 dimension */
	if ((vinput != NULL) && (vinput != Py_None)){
		*valuesArray = (PyArrayObject *)
    				PyArray_FROMANY(vinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
		if (*valuesArray == NULL)
		{
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			PyErr_SetString(st->error, "Values array cannot be converted to a float array.");
			return 0;
		}
		/* check the values array size */
		*vSize = 1;
		for (i=0; i< (*valuesArray)->nd;i++){
			*vSize *= (*valuesArray)->dimensions[i];
		}

		if (*vSize != (*xSize) * (*ySize) *  (*zSize)){
			Py_DECREF(*xArray);
			Py_DECREF(*yArray);
			Py_DECREF(*zArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			Py_DECREF(*valuesArray);
			PyErr_SetString(st->error, "Number of values does not match number of vertices.");
			return 0;
		}
	}

	*colorFilterFlag = cfilter;
	*valueFilterFlag = vfilter;
	*vMin = vmin;
	*vMax = vmax;

	return 1;
}


static int checkXYZVertexAndColor(PyObject *self, PyObject *args, PyArrayObject **xyzArray,\
						 PyArrayObject **colorArray,\
						 PyArrayObject **valuesArray,\
						 PyArrayObject **facetsArray,\
						 int *colorFilterFlag, int *valueFilterFlag, float *vMin, float *vMax,\
						 npy_intp *xyzSize, npy_intp *cSize, npy_intp *vSize, npy_intp *fSize)
{

	/* input parameters */
	PyObject	   *xyzinput, *cinput=NULL, *vinput=NULL, *finput=NULL;
	int			   cfilter=0, vfilter=0;
	float		   vmin=1, vmax=0;

	/* local variables */
	npy_intp	    i;
	struct module_state *st = GETSTATE(self);

	/* statements */
	if (!PyArg_ParseTuple(args, "O|OOOi(iff)", &xyzinput, &cinput, &vinput, &finput, &cfilter, &vfilter, &vmin, &vmax))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. At least three float arrays required");
        return 0;
	}

	/* convert to a contiguous array of at least 1 dimension */
	*xyzArray = (PyArrayObject *)
    				PyArray_FROMANY(xyzinput, NPY_FLOAT, 2, 2, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (*xyzArray == NULL)
	{
	    PyErr_SetString(st->error, "First argument cannot be converted to a three-columns float array.");
        return 0;
	}

	/* check the size of the vertex array */
	*xyzSize = (*xyzArray)->dimensions[0];
	if ((*xyzArray)->dimensions[1] != 3){
	    PyErr_SetString(st->error, "First argument cannot be converted to a three-columns float array.");
		Py_DECREF(*xyzArray);
        return 0;
	}


	/* convert to a contiguous array of at least 1 dimension */
	if ((cinput != NULL) && (cinput != Py_None)){
		*colorArray = (PyArrayObject *)
    					PyArray_ContiguousFromAny(cinput, NPY_UBYTE, 1, 0);
		if (*colorArray == NULL)
		{
			Py_DECREF(*xyzArray);
			PyErr_SetString(st->error, "Second argument cannot be converted to an unsigned byte array.");
			return 0;
		}
	}

	if ((cinput != NULL) && (cinput != Py_None)){
		*cSize = 1;
		for (i=0; i< (*colorArray)->nd;i++){
			*cSize *= (*colorArray)->dimensions[i];
		}

		if (*cSize != (4 * (*xyzSize))){
				Py_DECREF(*xyzArray);
				Py_DECREF(*colorArray);
				PyErr_SetString(st->error, "Number of colors does not match number of vertices.");
				return 0;
		}
	}

	/* convert to a contiguous array of at least 1 dimension */
	if ((vinput != NULL) && (vinput != Py_None)){
		*valuesArray = (PyArrayObject *)
    				PyArray_FROMANY(vinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
		if (*valuesArray == NULL)
		{
			Py_DECREF(*xyzArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			PyErr_SetString(st->error, "Values array cannot be converted to a float array.");
			return 0;
		}
		/* check the values array size */
		*vSize = 1;
		for (i=0; i< (*valuesArray)->nd;i++){
			*vSize *= (*valuesArray)->dimensions[i];
		}

		if (*vSize != (*xyzSize)){
			Py_DECREF(*xyzArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			Py_DECREF(*valuesArray);
			PyErr_SetString(st->error, "Number of values does not match number of vertices.");
			return 0;
		}
	}

	/* convert to a contiguous array of two dimensions */
	if ((finput != NULL) && (finput != Py_None)){
		*facetsArray = (PyArrayObject *)
    				PyArray_FROMANY(finput, NPY_UINT32, 2, 2, NPY_C_CONTIGUOUS|NPY_FORCECAST);
		if (*facetsArray == NULL)
		{
			Py_DECREF(*xyzArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			Py_DECREF(*valuesArray);
			PyErr_SetString(st->error, "Facets cannot be converted to an int32 array.");
			return 0;
		}
		/* check the facets array size */
		*fSize = (*facetsArray)->dimensions[0];
		if ((*facetsArray)->dimensions[1] != 3){
			PyErr_SetString(st->error, "Fourth argument cannot be converted to a three-columns float array.");
			Py_DECREF(*xyzArray);
			if ((cinput != NULL) && (cinput != Py_None))
			{
				Py_DECREF(*colorArray);
			}
			if ((vinput != NULL) && (vinput != Py_None))
			{
				Py_DECREF(*valuesArray);
			}
			Py_DECREF(*facetsArray);
			return 0;
		}
	}

	*colorFilterFlag = cfilter;
	*valueFilterFlag = vfilter;
	*vMin = vmin;
	*vMax = vmax;

	return 1;
}

static PyObject *draw2DGridPoints(PyObject *self, PyObject *args)
{

	PyArrayObject	*xArray, *yArray, *zArray;
	PyArrayObject	*colorArray;
	PyArrayObject   *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;
    /*struct module_state *st = GETSTATE(self);*/

	j = check2DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
	}

	/* The actual openGL stuff */
	pz = (float *) zArray->data;
	glBegin(GL_POINTS);
	if (pc == NULL){
		if ((vSize >0) && (vFilter != 0)){
			px = (float *) xArray->data;
			pv = (float *) valuesArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						glVertex3f(*px, *py, *pz);
					}
					pv++;
					pz++;
					pc += 4;
					py++;
				}
				px++;
			}
		}else{
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					glVertex3f(*px, *py, *pz);
					pz++;
					py++;
				}
				px++;
			}
		}
	}else{
		if (cFilter == 1){
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
					}else{
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
					}
					pz++;
					pc += 4;
					py++;
				}
				px++;
			}
		}else if ((vSize >0) && (vFilter != 0)){
			px = (float *) xArray->data;
			pv = (float *) valuesArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
					}
					pv++;
					pz++;
					pc += 4;
					py++;
				}
				px++;
			}
		}else{
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					glColor4ubv(pc);
					glVertex3f(*px, *py, *pz);
					pz++;
					pc += 4;
					py++;
				}
				px++;
			}
		}
	}
	glEnd();

	/* OpenGL stuff finished */

	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize >0){
		Py_DECREF(valuesArray);
	}
	Py_INCREF(Py_None);
	return(Py_None);

}

static PyObject *draw2DGridLines(PyObject *self, PyObject *args)
{
	PyArrayObject	*xArray, *yArray, *zArray;
	PyArrayObject	*colorArray;
	PyArrayObject   *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j, k, cidx;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;
    /*struct module_state *st = GETSTATE(self);*/

	/* statements */
	j = check2DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
	}
	/* The actual openGL stuff */
	if (pc == NULL){
		/* lines perpendicular to x axis direction */
		if ((vSize >0) && (vFilter != 0)){
			/* lines perpendicular to x axis direction */
			px = (float *) xArray->data;
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						glVertex3f(*px, *py, *pz);
					}
					pv++;
					pz++;
					py++;
				}
				glEnd();
				px++;
			}

			/* lines perpendicular to y axis direction */
			py = (float *) yArray->data;
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					k = i * ySize + j;
					if ((*(pv+k) < vMin) || (*(pv+k) > vMax)){
						/* do not plot */
					}else{
						glVertex3f(*px, *py, *(pz+k));
					}
					px++;
				}
				glEnd();
				py++;
			}
		}else{
			/* lines perpendicular to x axis direction */
			px = (float *) xArray->data;
			pz = (float *) zArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					glVertex3f(*px, *py, *pz);
					pz++;
					py++;
				}
				glEnd();
				px++;
			}

			/* lines perpendicular to y axis direction */
			py = (float *) yArray->data;
			pz = (float *) zArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					k = i * ySize + j;
					glVertex3f(*px, *py, *(pz+k));
					px++;
				}
				glEnd();
				py++;
			}
		}
	}else{
		if (cFilter == 1){
			/* lines perpendicular to x axis direction */
			px = (float *) xArray->data;
			pz = (float *) zArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						glEnd();
						glBegin(GL_LINE_STRIP);
					}else{
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
					}
					pc +=4;
					pz++;
					py++;
				}
				glEnd();
				px++;
			}

			/* lines perpendicular to y axis direction */
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			pz = (float *) zArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					k = i * ySize + j;
					cidx = 4 * k;
					pc += cidx;
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						glEnd();
						glBegin(GL_LINE_STRIP);
					}else{
						glColor4ubv(pc);
						glVertex3f(*px, *py, *(pz+k));
					}
					pc -=cidx;
					px++;
				}
				glEnd();
				py++;
			}
		}else if ((vSize >0) && (vFilter != 0)){
			/* lines perpendicular to x axis direction */
			px = (float *) xArray->data;
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_LINE_STRIP);
					}else{
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
					}
					pc +=4;
					pv++;
					pz++;
					py++;
				}
				glEnd();
				px++;
			}

			/* lines perpendicular to y axis direction */
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					k = i * ySize + j;
					cidx = 4 * k;
					if ((*(pv+k) < vMin) || (*(pv+k) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_LINE_STRIP);
					}else{
						glColor4ubv((pc+cidx));
						glVertex3f(*px, *py, *(pz+k));
					}
					px++;
				}
				glEnd();
				py++;
			}
		}else{
			px = (float *) xArray->data;
			pz = (float *) zArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					glColor4ubv(pc);
					glVertex3f(*px, *py, *pz);
					pc +=4;
					pz++;
					py++;
				}
				glEnd();
				px++;
			}

			/* lines perpendicular to y axis direction */
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			pz = (float *) zArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					k = i * ySize + j;
					cidx = 4 * k;
					glColor4ubv((pc+cidx));
					glVertex3f(*px, *py, *(pz+k));
					px++;
				}
				glEnd();
				py++;
			}
		}
	}

	/* OpenGL stuff finished */

	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize >0){
		Py_DECREF(valuesArray);
	}
	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *draw2DGridQuads(PyObject *self, PyObject *args)
{
	PyArrayObject	*xArray, *yArray, *zArray;
	PyArrayObject	*colorArray;
	PyArrayObject   *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j, k;
	npy_intp		cidx, coffset;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;
    /*struct module_state *st = GETSTATE(self);*/

	/* statements */
	j = check2DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
	}
	/* The actual openGL stuff */
	if (pc == NULL){
		if ((vSize >0) && (vFilter != 0)){
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pv = (float *) valuesArray->data;
			pz = (float *) zArray->data;
			px = (float *) xArray->data;
			for (i=0; i< (xSize-1); i++){
				py = (float *) yArray->data;
				for (j=0; j<(ySize-1); j++){
					/* face x0y0 */
					k = i * ySize + j;
					if ((*(pv+k) < vMin) || (*(pv+k) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glVertex3f(*px, *py, *(pz+k));
					/* face x1y0 */
					if ((*(pv+(k+ySize)) < vMin) || (*(pv+(k+ySize)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glVertex3f(*(px+1), *py, *(pz+(k+ySize)));
					/* face x1y1 */
					if ((*(pv+(k+ySize+1)) < vMin) || (*(pv+(k+ySize+1)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glVertex3f(*(px+1), *(py+1), *(pz+(k+ySize+1)));
					/* face x0y1 */
					if ((*(pv+(k+1)) < vMin) || (*(pv+(k+1)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glVertex3f(*(px), *(py+1), *(pz+(k+1)));
					py++;
				}
				px++;
			}
			glEnd();
		}else{
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			px = (float *) xArray->data;
			for (i=0; i< (xSize-1); i++){
				py = (float *) yArray->data;
				for (j=0; j<(ySize-1); j++){
					/* face x0y0 */
					k = i * ySize + j;
					glVertex3f(*px, *py, *(pz+k));
					/* face x1y0 */
					glVertex3f(*(px+1), *py, *(pz+(k+ySize)));
					/* face x1y1 */
					glVertex3f(*(px+1), *(py+1), *(pz+(k+ySize+1)));
					/* face x0y1 */
					glVertex3f(*(px), *(py+1), *(pz+(k+1)));
					py++;
				}
				px++;
			}
			glEnd();
		}
	}else{
		if (cFilter == 1){
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			px = (float *) xArray->data;
			for (i=0; i< (xSize-1); i++){
				py = (float *) yArray->data;
				for (j=0; j<(ySize-1); j++){
					/* face x0y0 */
					k = i * ySize + j;
					coffset = 4 * k;
					cidx = coffset;
					pc += cidx;
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						pc -= cidx;
						py ++;
						glEnd();
						glBegin(GL_QUADS);
						continue;
					}
					glColor4ubv(pc);
					glVertex3f(*px, *py, *(pz+k));
					pc -= cidx;
					/* face x1y0 */
					cidx = coffset+4*ySize;
					pc += cidx;
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						pc -= cidx;
						py ++;
						glEnd();
						glBegin(GL_QUADS);
						continue;
					}
					glColor4ubv(pc);
					glVertex3f(*(px+1), *py, *(pz+(k+ySize)));
					pc -= cidx;
					/* face x1y1 */
					cidx = coffset+ 4*ySize+4;
					pc += cidx;
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						pc -= cidx;
						py ++;
						glEnd();
						glBegin(GL_QUADS);
						continue;
					}
					glColor4ubv(pc);
					glVertex3f(*(px+1), *(py+1), *(pz+(k+ySize+1)));
					pc -= cidx;
					/* face x0y1 */
					cidx = coffset + 4;
					pc += cidx;
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
						pc -= cidx;
						py ++;
						glEnd();
						glBegin(GL_QUADS);
						continue;
					}
					glColor4ubv(pc);
					glVertex3f(*(px), *(py+1), *(pz+(k+1)));
					pc -= cidx;
					py++;
				}
				px++;
			}
			glEnd();
		}else if ((vSize >0) && (vFilter != 0)){
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pv = (float *) valuesArray->data;
			pz = (float *) zArray->data;
			px = (float *) xArray->data;
			for (i=0; i< (xSize-1); i++){
				py = (float *) yArray->data;
				for (j=0; j<(ySize-1); j++){
					/* face x0y0 */
					k = i * ySize + j;
					coffset = 4 * k;
					cidx = coffset;
					if ((*(pv+k) < vMin) || (*(pv+k) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glColor4ubv((pc+cidx));
					glVertex3f(*px, *py, *(pz+k));
					/* face x1y0 */
					cidx = coffset+4*ySize;
					if ((*(pv+(k+ySize)) < vMin) || (*(pv+(k+ySize)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glColor4ubv((pc+cidx));
					glVertex3f(*(px+1), *py, *(pz+(k+ySize)));
					/* face x1y1 */
					cidx = coffset+ 4*ySize+4;
					if ((*(pv+(k+ySize+1)) < vMin) || (*(pv+(k+ySize+1)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					glColor4ubv((pc+cidx));
					glVertex3f(*(px+1), *(py+1), *(pz+(k+ySize+1)));
					/* face x0y1 */
					if ((*(pv+(k+1)) < vMin) || (*(pv+(k+1)) > vMax)){
						/* do not plot */
						glEnd();
						glBegin(GL_QUADS);
						py++;
						continue;
					}
					cidx = coffset + 4;
					glColor4ubv((pc+cidx));
					glVertex3f(*(px), *(py+1), *(pz+(k+1)));
					py++;
				}
				px++;
			}
			glEnd();
		}else{
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			px = (float *) xArray->data;
			for (i=0; i< (xSize-1); i++){
				py = (float *) yArray->data;
				for (j=0; j<(ySize-1); j++){
					/* face x0y0 */
					k = i * ySize + j;
					coffset = 4 * k;
					cidx = coffset;
					glColor4ubv((pc+cidx));
					glVertex3f(*px, *py, *(pz+k));
					/* face x1y0 */
					cidx = coffset+4*ySize;
					glColor4ubv((pc+cidx));
					glVertex3f(*(px+1), *py, *(pz+(k+ySize)));
					/* face x1y1 */
					cidx = coffset+ 4*ySize+4;
					glColor4ubv((pc+cidx));
					glVertex3f(*(px+1), *(py+1), *(pz+(k+ySize+1)));
					/* face x0y1 */
					cidx = coffset + 4;
					glColor4ubv((pc+cidx));
					glVertex3f(*(px), *(py+1), *(pz+(k+1)));
					py++;
				}
				px++;
			}
			glEnd();
		}
	}

	/* OpenGL stuff finished */

	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize >0){
		Py_DECREF(valuesArray);
	}
	Py_INCREF(Py_None);
	return(Py_None);
}


static PyObject *draw3DGridPoints(PyObject *self, PyObject *args)
{
	PyArrayObject	*xArray, *yArray, *zArray, *colorArray, *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j, k;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;
	GLuint			index, indexOffset=0;
	typedef struct VertexArray {
		GLfloat     x;
		GLfloat     y;
		GLfloat     z;
	}VertexArray;

	typedef struct ColorArray{
		GLubyte		r;
		GLubyte     g;
		GLubyte     b;
		GLubyte     a;
	}ColorArray;

	VertexArray		*vertexArrayGL=NULL, *pVAGL;
	ColorArray		*colorArrayGL=NULL,  *pCAGL;
	GLuint			*indexArray=NULL,  *pIA;

	GLint			maxElementsVertices=4096; /* 256 * 256 * 256 in my card */
	GLint			maxElementsIndices=4096;  /* 65535 in my card */
	GLint			maxElements=4096;

	j = check3DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	/* The actual openGL stuff */
	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &maxElementsVertices);
	if (maxElementsVertices <= 0)
	{	maxElementsVertices = 1000;
		printf("Max elements vertices <= 0, forced to 1000\n");
	}
	glGetIntegerv(GL_MAX_ELEMENTS_INDICES,  &maxElementsIndices);
	if (maxElementsIndices <= 0)
	{	maxElementsIndices = 1000;
		printf("Max elements vertices <= 0, forced to 1000\n");
	}
	if (maxElementsVertices > (xSize * ySize *zSize))
	{
		maxElementsVertices = xSize * ySize *zSize;
	}
	if (maxElementsIndices > (xSize * ySize *zSize))
	{
		maxElementsIndices = xSize * ySize * zSize;
	}
	if (maxElementsIndices < maxElementsVertices)
	{
		maxElements = maxElementsIndices;
	}else{
		maxElements = maxElementsVertices;
	}
	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
		colorArrayGL  = (ColorArray *) malloc(maxElements * sizeof(ColorArray));
	}
	indexArray  = (GLuint *) malloc(maxElements * sizeof(GLuint));
	vertexArrayGL = (VertexArray *) malloc(maxElements * sizeof(VertexArray));
	/* The actual openGL stuff */
	if (pc == NULL){
		if ((vSize >0) && (vFilter != 0)){
			glBegin(GL_POINTS);
			pv = (float *) valuesArray->data;
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if ((*pv < vMin) || (*pv > vMax)){
							/* do not plot */
						}else{
							glVertex3f(*px, *py, *pz);
						}
						pv++;
						pz++;
						pc += 4;
					}
					py++;
				}
				px++;
			}
			glEnd();
		}else{
			if ((indexArray != NULL) && (vertexArrayGL != NULL))
			{
				pVAGL = vertexArrayGL;
				glVertexPointer(3, GL_FLOAT, 0, vertexArrayGL);
				glEnableClientState(GL_VERTEX_ARRAY);
				px = (float *) xArray->data;
				index=0;
				for (i=0; i<xSize; i++){
					py = (float *) yArray->data;
					for (j=0; j<ySize; j++){
						pz = (float *) zArray->data;
						for (k=0; k<zSize; k++){
							pVAGL->x = *px;
							pVAGL->y = *py;
							pVAGL->z = *pz;
							index++;
							if (index == maxElements){
								pIA = indexArray;
								pVAGL = vertexArrayGL;
								glDrawArrays(GL_POINTS, 0, index);
								indexOffset+=index;
								index=0;
							}else{
								pIA++;
								pVAGL++;
							}
							pz++;
						}
						py++;
					}
					px++;
				}
				if(index > 0)
				{
					glDrawArrays(GL_POINTS, 0, index);
				}
				glDisableClientState(GL_VERTEX_ARRAY);
			}else{
				glBegin(GL_POINTS);
				px = (float *) xArray->data;
				for (i=0; i<xSize; i++){
					py = (float *) yArray->data;
					for (j=0; j<ySize; j++){
						pz = (float *) zArray->data;
						for (k=0; k<zSize; k++){
							glVertex3f(*px, *py, *pz);
							pz++;
						}
						py++;
					}
					px++;
				}
				glEnd();
			}
		}
	}else{
		if (cFilter == 1){
			glBegin(GL_POINTS);
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
						}else{
							glColor4ubv(pc);
							glVertex3f(*px, *py, *pz);
						}
						pz++;
						pc += 4;
					}
					py++;
				}
				px++;
			}
			glEnd();
		}else if ((vSize >0) && (vFilter != 0)){
			glBegin(GL_POINTS);
			pv = (float *) valuesArray->data;
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if ((*pv < vMin) || (*pv > vMax)){
							/* do not plot */
						}else{
							glColor4ubv(pc);
							glVertex3f(*px, *py, *pz);
						}
						pv++;
						pz++;
						pc += 4;
					}
					py++;
				}
				px++;
			}
			glEnd();
		}else if (0 && (indexArray != NULL) && (vertexArrayGL != NULL) && (colorArrayGL != NULL)){
			pIA = indexArray;
			pVAGL = vertexArrayGL;
			pCAGL = colorArrayGL;
			glVertexPointer(3, GL_FLOAT, 0, vertexArrayGL);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, colorArrayGL);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			px = (float *) xArray->data;
			index=0;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						pVAGL->x = *px;
						pVAGL->y = *py;
						pVAGL->z = *pz;
						pCAGL->r = *pc;
						pc++;
						pCAGL->g = *pc;
						pc++;
						pCAGL->b = *pc;
						pc++;
						pCAGL->a = *pc;
						pc++;
						*pIA = index;
						index++;
						if (index == maxElements){
							index = 0;
							pIA = indexArray;
							pVAGL = vertexArrayGL;
							pCAGL = colorArrayGL;
							/*
							glDrawRangeElements(GL_POINTS,\
								0, maxElements-1, maxElements,\
								GL_UNSIGNED_INT, pIA);
							*/
							glDrawElements(GL_POINTS,\
									maxElements,\
									GL_UNSIGNED_INT, pIA);
						}else{
							pIA++;
							pVAGL++;
							pCAGL++;
						}
						pz++;
					}
					py++;
				}
				px++;
			}
			if(index > 0)
			{
				glDrawElements(GL_POINTS,index,\
							GL_UNSIGNED_INT, pIA);
			}
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}else if (1 && (indexArray != NULL) && (vertexArrayGL != NULL) && (colorArrayGL != NULL)){
			pVAGL = vertexArrayGL;
			glVertexPointer(3, GL_FLOAT, 0, vertexArrayGL);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, colorArrayGL);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			px = (float *) xArray->data;
			index=0;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						pVAGL->x = *px;
						pVAGL->y = *py;
						pVAGL->z = *pz;
						index++;
						if (index == maxElements){
							pIA = indexArray;
							pVAGL = vertexArrayGL;
							memcpy(colorArrayGL, (pc+(4*indexOffset)), (4*index));
							glDrawArrays(GL_POINTS, 0, index);
							indexOffset+=index;
							index=0;
						}else{
							pIA++;
							pVAGL++;
						}
						pz++;
					}
					py++;
				}
				px++;
			}
			if(index > 0)
			{
				memcpy(colorArrayGL, (pc+(4*indexOffset)), (4*index));
				glDrawArrays(GL_POINTS, 0, index);
			}
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}else{
			glBegin(GL_POINTS);
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
						pz++;
						pc += 4;
					}
					py++;
				}
				px++;
			}
			glEnd();
		}
	}

	/* OpenGL stuff finished */

	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (indexArray != NULL){
		free(indexArray);
	}
	if (vertexArrayGL != NULL){
		free(vertexArrayGL);
	}
	if (colorArrayGL != NULL){
		free(colorArrayGL);
	}
	Py_INCREF(Py_None);
	return(Py_None);

}

static PyObject *draw3DGridLines(PyObject *self, PyObject *args)
{
	PyArrayObject	*xArray, *yArray, *zArray, *colorArray, *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j, k, cidx, vidx;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;

	/* statements */
	j = check3DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
	}

	/* The actual openGL stuff */
	if (pc == NULL){
		if ((vSize >0) && (vFilter != 0)){
			pv = (float *) valuesArray->data;
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if ((*pv < vMin) || (*pv > vMax)){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							glVertex3f(*px, *py, *pz);
						}
						pv++;
						pz++;
					}
					py++;
				}
				glEnd();
				px++;
			}
			/* lines perpendicular to y axis direction */
			pv = (float *) valuesArray->data;
			py = (float *) yArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					pz = (float *) zArray->data;
					vidx = (i * ySize + j) * zSize;
					for (k=0; k<zSize; k++){
						vidx += k;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							glVertex3f(*px, *py, *pz);
						}
						pz++;
					}
					px++;
				}
				glEnd();
				py++;
			}
		}else{
			/* lines perpendicular to x axis direction */
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						glVertex3f(*px, *py, *pz);
						pz++;
					}
					py++;
				}
				glEnd();
				px++;
			}
			/* lines perpendicular to y axis direction */
			py = (float *) yArray->data;
			for (i=0; i<ySize; i++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<xSize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						glVertex3f(*px, *py, *pz);
						pz++;
					}
					px++;
				}
				glEnd();
				py++;
		}
 }
	}else{
		/* lines perpendicular to x axis direction */
		if (cFilter == 1){
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							glColor4ubv(pc);
							glVertex3f(*px, *py, *pz);
						}
						pc += 4;
						pz++;
					}
					py++;
				}
				glEnd();
				px++;
			}
			/* lines perpendicular to y axis direction */
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						cidx = 4 * ((i * ySize + j) * zSize + k);
						pc += cidx;
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							glColor4ubv(pc);
							glVertex3f(*px, *py, *pz);
						}
						pc -= cidx;
						pz++;
					}
					px++;
				}
				glEnd();
				py++;
			}
		}else if ((vSize >0) && (vFilter != 0)){
			pv = (float *) valuesArray->data;
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						if ((*pv < vMin) || (*pv > vMax)){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							glColor4ubv(pc);
							glVertex3f(*px, *py, *pz);
						}
						pv++;
						pc += 4;
						pz++;
					}
					py++;
				}
				glEnd();
				px++;
			}
			/* lines perpendicular to y axis direction */
			pv = (float *) valuesArray->data;
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						vidx = ((i * ySize + j) * zSize + k);
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							glEnd();
							glBegin(GL_LINE_STRIP);
						}else{
							cidx = 4 * vidx;
							glColor4ubv((pc+cidx));
							glVertex3f(*px, *py, *pz);
						}
						pz++;
					}
					px++;
				}
				glEnd();
				py++;
			}
		}else{
			px = (float *) xArray->data;
			for (i=0; i<xSize; i++){
				py = (float *) yArray->data;
				glBegin(GL_LINE_STRIP);
				for (j=0; j<ySize; j++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
						pc += 4;
						pz++;
					}
					py++;
				}
				glEnd();
				px++;
			}
			/* lines perpendicular to y axis direction */
			pc = (GLubyte *) colorArray->data;
			py = (float *) yArray->data;
			for (j=0; j<ySize; j++){
				px = (float *) xArray->data;
				glBegin(GL_LINE_STRIP);
				for (i=0; i<xSize; i++){
					pz = (float *) zArray->data;
					for (k=0; k<zSize; k++){
						cidx = 4 * ((i * ySize + j) * zSize + k);
						glColor4ubv((pc+cidx));
						glVertex3f(*px, *py, *pz);
						pz++;
					}
					px++;
				}
				glEnd();
				py++;
			}
		}
	}

	/* OpenGL stuff finished */

	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *draw3DGridQuads(PyObject *self, PyObject *args)
{
	PyArrayObject	*xArray, *yArray, *zArray, *colorArray, *valuesArray;
	npy_intp	    xSize, ySize, zSize, cSize=0, vSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j, k;
	npy_intp		cidx, vidx, deltaX, coffset;
	float			*px, *py, *pz, *pv;
	GLubyte			*pc=NULL;

	/* statements */
	j = check3DGridVertexAndColor(self, args, &xArray, &yArray, &zArray, &colorArray,\
							&valuesArray, &cFilter, &vFilter, &vMin, &vMax, &xSize, &ySize, &zSize, &cSize, &vSize);
	if (!j)
		return NULL;

	if (cSize > 0)
	{
		pc = (GLubyte *) colorArray->data;
	}
	/* The actual openGL stuff */
	if (pc == NULL){
		if ((vSize >0) && (vFilter != 0)){
			deltaX = ySize * zSize;
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (k=0; k<zSize; k++){
				px = (float *) xArray->data;
				for (i=0; i< (xSize-1); i++){
					py = (float *) yArray->data;
					for (j=0; j<(ySize-1); j++){
						/* test vertex x0y0 */
						vidx = (i * ySize + j) * zSize + k;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}

						/* test vertex x0y1 */
						vidx += zSize;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}

						/* test vertex x1y0 */
						vidx = ((i+1) * ySize + j) * zSize + k;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}

						/* test vertex x1y1 */
						vidx += zSize;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}

						/* vertex x0y0 */
						glVertex3f(*px, *py, *pz);
						/* vertex x1y0 */
						glVertex3f(*(px+1), *py, *pz);
						/* vertex x1y1 */
						glVertex3f(*(px+1), *(py+1), *pz);
						/* vertex x0y1 */
						glVertex3f(*(px), *(py+1), *pz);
						py++;
					}
					px++;
				}
				pz++;
			}
			glEnd();
		}else{
			deltaX = ySize * zSize;
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			for (k=0; k<zSize; k++){
				px = (float *) xArray->data;
				for (i=0; i< (xSize-1); i++){
					py = (float *) yArray->data;
					for (j=0; j<(ySize-1); j++){
						/* face x0y0 */
						glVertex3f(*px, *py, *pz);
						/* face x1y0 */
						glVertex3f(*(px+1), *py, *pz);
						/* face x1y1 */
						glVertex3f(*(px+1), *(py+1), *pz);
						/* face x0y1 */
						glVertex3f(*(px), *(py+1), *pz);
						py++;
					}
					px++;
				}
				pz++;
			}
			glEnd();
		}
	}else{
		if (cFilter == 1){
			deltaX = ySize * zSize;
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			for (k=0; k<zSize; k++){
				px = (float *) xArray->data;
				for (i=0; i< (xSize-1); i++){
					py = (float *) yArray->data;
					for (j=0; j<(ySize-1); j++){
						pc = (GLubyte *) colorArray->data;
						/* vertex x0y0 */
						coffset = 4 * (i * deltaX + j * zSize + k);
						cidx = coffset;
						pc += cidx;
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_QUADS);
							py++;
							continue;
						}
						glColor4ubv(pc);
						glVertex3f(*px, *py, *pz);
						pc -= cidx,

						/* vertex x1y0 */
						cidx = coffset+4*deltaX;
						pc += cidx;
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_QUADS);
							py++;
							continue;
						}
						glColor4ubv(pc);
						glVertex3f(*(px+1), *py, *pz);
						pc -= cidx,

						/* vertex x1y1 */
						cidx = coffset+ 4*deltaX + 4*zSize;
						pc += cidx;
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_QUADS);
							py++;
							continue;
						}
						glColor4ubv(pc);
						glVertex3f(*(px+1), *(py+1), *pz);
						pc -= cidx,

						/* vertex x0y1 */
						cidx = coffset + 4*zSize;
						pc += cidx;
						if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
						    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
							glEnd();
							glBegin(GL_QUADS);
							py++;
							continue;
						}
						glColor4ubv(pc);
						glVertex3f(*(px), *(py+1), *pz);
						pc -= cidx,
						py++;
					}
					px++;
				}
				pz++;
			}
			glEnd();
		}else if ((vSize >0) && (vFilter != 0)){
			deltaX = ySize * zSize;
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			pv = (float *) valuesArray->data;
			for (k=0; k<zSize; k++){
				px = (float *) xArray->data;
				for (i=0; i< (xSize-1); i++){
					py = (float *) yArray->data;
					for (j=0; j<(ySize-1); j++){
						/* test vertex x0y0 */
						vidx = (i * ySize + j) * zSize + k;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}
						/* test vertex x0y1 */
						vidx += zSize;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}
						/* test vertex x1y0 */
						vidx = ((i+1) * ySize + j) * zSize + k;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}
						/* test vertex x1y1 */
						vidx += zSize;
						if ((*(pv+vidx) < vMin) || (*(pv+vidx) > vMax)){
							/* do not plot */
							py++;
							continue;
						}
						/* vertex x0y0 */
						coffset = 4 * (i * deltaX + j * zSize + k);
						cidx = coffset;
						glColor4ubv((pc+cidx));
						glVertex3f(*px, *py, *pz);
						/* vertex x1y0 */
						cidx = coffset+4*deltaX;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px+1), *py, *pz);
						/* vertex x1y1 */
						cidx = coffset+ 4*deltaX + 4*zSize;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px+1), *(py+1), *pz);
						/* vertex x0y1 */
						cidx = coffset + 4*zSize;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px), *(py+1), *pz);
						py++;
					}
					px++;
				}
				pz++;
			}
			glEnd();
		}else{
			deltaX = ySize * zSize;
			glBegin(GL_QUADS);
			/* Quads in the XY planes */
			pz = (float *) zArray->data;
			for (k=0; k<zSize; k++){
				px = (float *) xArray->data;
				for (i=0; i< (xSize-1); i++){
					py = (float *) yArray->data;
					for (j=0; j<(ySize-1); j++){
						/* vertex x0y0 */
						coffset = 4 * (i * deltaX + j * zSize + k);
						cidx = coffset;
						glColor4ubv((pc+cidx));
						glVertex3f(*px, *py, *pz);
						/* vertex x1y0 */
						cidx = coffset+4*deltaX;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px+1), *py, *pz);
						/* vertex x1y1 */
						cidx = coffset+ 4*deltaX + 4*zSize;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px+1), *(py+1), *pz);
						/* vertex x0y1 */
						cidx = coffset + 4*zSize;
						glColor4ubv((pc+cidx));
						glVertex3f(*(px), *(py+1), *pz);
						py++;
					}
					px++;
				}
				pz++;
			}
			glEnd();
		}
	}

	/* OpenGL stuff finished */
	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *drawXYZPoints(PyObject *self, PyObject *args)
{
	PyArrayObject	*xyzArray, *colorArray, *valuesArray, *facetsArray;
	npy_intp	    xyzSize, cSize=0, vSize=0, fSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j;
	float			*pxyz, *pv;
	GLubyte			*pc=NULL;
	GLsizei			nVertices;

	/* statements */
	j = checkXYZVertexAndColor(self, args, &xyzArray, &colorArray, &valuesArray, &facetsArray,
					&cFilter, &vFilter, &vMin, &vMax, &xyzSize, &cSize, &vSize, &fSize);
	if (!j)
		return NULL;

	nVertices = xyzSize;
	pxyz = (float *) xyzArray->data;
	if (cSize > 0){
		pc = (GLubyte *) colorArray->data;
	}

	if (pc == NULL)	{
		if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			pv = (float *) valuesArray->data;
			glBegin(GL_POINTS);
			for (i=0; i< nVertices; i++){
				if ((*pv < vMin) || (*pv > vMax)){
					/* do not plot */
				}else{
					glVertex3fv(pxyz);
				}
				pxyz += 3;
				pv ++;
			}
			glEnd();
		}else{
			/*This case could be made in pure python */
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glEnableClientState(GL_VERTEX_ARRAY);
			glDrawArrays(GL_POINTS, 0, nVertices);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}else{
		if (cFilter == 1){
			/* We have to loop */
			glBegin(GL_POINTS);
			for (i=0; i< nVertices; i++){
				if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
				    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
						/* do not plot */
				}else{
					glColor4ubv(pc);
					glVertex3fv(pxyz);
				}
				pxyz += 3;
				pc += 4;
			}
			glEnd();
		}else if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			pv = (float *) valuesArray->data;
			glBegin(GL_POINTS);
			for (i=0; i< nVertices; i++){
				if ((*pv < vMin) || (*pv > vMax)){
					/* do not plot */
				}else{
					glColor4ubv(pc);
					glVertex3fv(pxyz);
				}
				pxyz += 3;
				pc += 4;
				pv ++;
			}
			glEnd();
		}else if (0){
			/* This is just for test */
			glBegin(GL_POINTS);
			for (i=0; i< nVertices; i++){
				glColor4ubv(pc);
				glVertex3fv(pxyz);
				pxyz += 3;
				pc += 4;
			}
			glEnd();
		}else{
			/* This case could be made in pure python */
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, pc);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			glDrawArrays(GL_POINTS, 0, nVertices);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}
	/* OpenGL stuff finished */
	Py_DECREF(xyzArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize != 0){
		Py_DECREF(valuesArray);
	}
	if (fSize != 0){
		Py_DECREF(facetsArray);
	}

	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *drawXYZLines(PyObject *self, PyObject *args)
{
	PyArrayObject	*xyzArray, *colorArray, *valuesArray, *facetsArray;
	npy_intp	    xyzSize, cSize=0, vSize=0, fSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j;
	float			*pxyz, *pv;
	GLubyte			*pc=NULL;
	unsigned int	*pf=NULL;
	GLsizei			facetDepth;

	/* statements */
	j = checkXYZVertexAndColor(self, args, &xyzArray, &colorArray, &valuesArray, &facetsArray,
					&cFilter, &vFilter, &vMin, &vMax, &xyzSize, &cSize, &vSize, &fSize);
	if (!j)
		return NULL;

	if (fSize == 0){
		/* Nothing to be made */
		printf("Warning: No facets to be drawn\n");
		Py_DECREF(xyzArray);
		if (cSize != 0){
			Py_DECREF(colorArray);
		}
		if (vSize != 0){
			Py_DECREF(valuesArray);
		}
		Py_INCREF(Py_None);
		return(Py_None);
	}

	pxyz = (float *) xyzArray->data;
	if (cSize > 0){
		pc = (GLubyte *) colorArray->data;
	}
	pf = (unsigned int *) facetsArray->data;

	if (pc == NULL)	{
		if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_LINE_LOOP);
				for (j=0; j<facetDepth; j++){
					pv = (float *) valuesArray->data;
					pv += *pf;
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else{
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glEnableClientState(GL_VERTEX_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i < fSize; i++){
	            glDrawElements(GL_LINE_LOOP,
                               facetDepth,
                               GL_UNSIGNED_INT,
                               pf);
			    pf += facetDepth;
			}
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}else{
		if (cFilter == 1){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_LINE_LOOP);
				for (j=0; j<facetDepth; j++){
					pc = (GLubyte *) colorArray->data;
					pc += 4*(*pf);
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						glColor4ubv(pc);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_LINE_LOOP);
				for (j=0; j<facetDepth; j++){
					/* incrementing the pointer does not work, perhaps my array is not contiguous?
						printf("This does not work and I do not know why\n");
						pxyz = (float *) xyzArray->data;
						pxyz += *pf;
						pc = (GLubyte *) colorArray->data;
						pc += 4*(*pf);
						if (i<2)
						{
							printf("i = %d, j=%d, facet = %u, vertex = %f, %f, %f\n", i, j, *pf, *pxyz, *(pxyz+1), *(pxyz+2));
							printf("i = %d, j=%d, facet = %u, color = %d, %d, %d, %d\n", i, j, *pf, *pc, *(pc+1), *(pc+2), *(pc+3));
						}
					*/
					pv = (float *) valuesArray->data;
					pv += *pf;
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						pc = (GLubyte *) colorArray->data;
						pc += 4*(*pf);
						glColor4ubv(pc);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else if (0){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, pc);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i < fSize; i++){
				glBegin(GL_LINE_LOOP);
				for (j=0; j<facetDepth; j++){
		            glArrayElement((GLint) *pf);
					pf++;
				}
				glEnd();
			}
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}else{
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, pc);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i < fSize; i++){
	            glDrawElements(GL_LINE_LOOP,
                               facetDepth,
                               GL_UNSIGNED_INT,
                               pf);
			    pf += facetDepth;
			}
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}

	/* OpenGL stuff finished */
	Py_DECREF(xyzArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize != 0){
		Py_DECREF(valuesArray);
	}
	if (fSize != 0){
		Py_DECREF(facetsArray);
	}

	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *drawXYZTriangles(PyObject *self, PyObject *args)
{
	PyArrayObject	*xyzArray, *colorArray, *valuesArray, *facetsArray;
	npy_intp	    xyzSize, cSize=0, vSize=0, fSize=0;
	int				cFilter=0, vFilter=0;
	float			vMin=1.0, vMax=0.0;
	npy_intp	    i, j;
	float			*pxyz, *pv;
	GLubyte			*pc=NULL;
	unsigned int	*pf=NULL;
	GLsizei			facetDepth;
    /*struct module_state *st = GETSTATE(self);*/

	/* statements */
	j = checkXYZVertexAndColor(self, args, &xyzArray, &colorArray, &valuesArray, &facetsArray,
					&cFilter, &vFilter, &vMin, &vMax, &xyzSize, &cSize, &vSize, &fSize);
	if (!j)
		return NULL;

	if (fSize == 0){
		/* Nothing to be made */
		printf("Warning: No facets to be drawn\n");
		Py_DECREF(xyzArray);
		if (cSize != 0){
			Py_DECREF(colorArray);
		}
		if (vSize != 0){
			Py_DECREF(valuesArray);
		}
		Py_INCREF(Py_None);
		return(Py_None);
	}

	pxyz = (float *) xyzArray->data;
	if (cSize > 0){
		pc = (GLubyte *) colorArray->data;
	}
	pf = (unsigned int *) facetsArray->data;

	if (pc == NULL)	{
		if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_TRIANGLES);
				for (j=0; j<facetDepth; j++){
					pv = (float *) valuesArray->data;
					pv += *pf;
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
						glBegin(GL_TRIANGLES);
						glEnd();
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else{
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glEnableClientState(GL_VERTEX_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
            glDrawElements(GL_TRIANGLES,
                           facetDepth * fSize,
                           GL_UNSIGNED_INT,
                           pf);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}else{
		if (cFilter == 1){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_TRIANGLES);
				for (j=0; j<facetDepth; j++){
					pc = (GLubyte *) colorArray->data;
					pc += 4*(*pf);
					if (((*pc == 255) && (*(pc+1) == 0) && (*(pc+2) == 0)) ||\
					    ((*pc == 0) && (*(pc+1) == 0) && (*(pc+2) == 255))){
							/* do not plot */
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						glColor4ubv(pc);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else if ((vSize >0) && (vFilter != 0)){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i<fSize; i++){
				glBegin(GL_TRIANGLES);
				for (j=0; j<facetDepth; j++){
					/* incrementing the pointer does not work, perhaps my array is not contiguous?
						printf("This does not work and I do not know why\n");
						pxyz = (float *) xyzArray->data;
						pxyz += *pf;
						pc = (GLubyte *) colorArray->data;
						pc += 4*(*pf);
						if (i<2)
						{
							printf("i = %d, j=%d, facet = %u, vertex = %f, %f, %f\n", i, j, *pf, *pxyz, *(pxyz+1), *(pxyz+2));
							printf("i = %d, j=%d, facet = %u, color = %d, %d, %d, %d\n", i, j, *pf, *pc, *(pc+1), *(pc+2), *(pc+3));
						}
					*/
					pv = (float *) valuesArray->data;
					pv += *pf;
					if ((*pv < vMin) || (*pv > vMax)){
						/* do not plot */
					}else{
						pxyz = (float *) PyArray_GETPTR2(xyzArray, *pf ,0);
						pc = (GLubyte *) colorArray->data;
						pc += 4*(*pf);
						glColor4ubv(pc);
						glVertex3fv(pxyz);
					}
					pf++;
				}
				glEnd();
			}
		}else if (0){
			/* We have to loop */
			facetDepth = (facetsArray)->dimensions[1];
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, pc);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
			for (i=0; i < fSize; i++){
				glBegin(GL_TRIANGLES);
				for (j=0; j<facetDepth; j++){
		            glArrayElement((GLint) *pf);
					pf++;
				}
				glEnd();
			}
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}else{
			glVertexPointer(3, GL_FLOAT, 0, pxyz);
			glColorPointer(4, GL_UNSIGNED_BYTE, 0, pc);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_COLOR_ARRAY);
			facetDepth = (facetsArray)->dimensions[1];
            glDrawElements(GL_TRIANGLES,
                               facetDepth * fSize,
                               GL_UNSIGNED_INT,
                               pf);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}

	/* OpenGL stuff finished */
	Py_DECREF(xyzArray);
	if (pc != NULL){
		Py_DECREF(colorArray);
	}
	if (vSize != 0){
		Py_DECREF(valuesArray);
	}
	if (fSize != 0){
		Py_DECREF(facetsArray);
	}

	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *getVertexArrayMeshAxes(PyObject *self, PyObject *args)
{
	PyObject		*inputArray;
	PyArrayObject	*vertexArray, *xArray, *yArray;
	npy_intp	    nVertices, xSize=0, ySize=0;
	npy_intp	    i, j, dim[1];
	float			*pv, delta=1.0E-8f;
	float			*xBuffer=NULL, *yBuffer=NULL;
	short			notAMesh=0, xRepeatedFirst=1;
    struct module_state *st = GETSTATE(self);
	/* statements */

	if (!PyArg_ParseTuple(args, "O|f", &inputArray, &delta))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. One float array required");
        return NULL;
	}

	/* convert to a contiguous array of at least 1 dimension */
	vertexArray = (PyArrayObject *)
    				PyArray_FROMANY(inputArray, NPY_FLOAT, 2, 2, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (vertexArray == NULL)
	{
	    PyErr_SetString(st->error, "Argument cannot be converted to an r x 3 float array.");
        return NULL;
	}

	if (vertexArray->nd != 2)
	{
	    PyErr_SetString(st->error, "Input array cannot be converted to an r x 3 float array.");
        return NULL;
	}

	nVertices = 1;
	for (i=0; i < vertexArray->nd;i++)
	{
		nVertices *= vertexArray->dimensions[i];
	}
	nVertices = (int)(nVertices/3);
	pv = (float *) vertexArray->data;
	ySize = 0;
	for (i=0; i<(nVertices-1); i++)
	{
		if (fabs(*pv - *(pv+i*3)) > delta)
		{
			break;
		}else{
			ySize += 1;
		}
	}

	pv++;
	xSize = 0;
	for (i=0; i<(nVertices-1); i++)
	{
		if (fabs(*pv - *(pv+i*3)) > delta)
		{
			break;
		}else{
			xSize += 1;
		}
	}

	if (ySize >= xSize)
	{
		xRepeatedFirst = 1;
		if ((ySize > 1) && (ySize < (nVertices-1)))
		{
			if (nVertices % ySize){
				/* not an integer */
				Py_DECREF(vertexArray);
				Py_INCREF(Py_None);
				return(Py_None);
			}else{
				xSize = nVertices/ySize;
			}
		}else{
				Py_DECREF(vertexArray);
				Py_INCREF(Py_None);
				return(Py_None);
		}
	}else{
		xRepeatedFirst = 0;
		if ((xSize > 1) && (xSize < (nVertices-1)))
		{
			if (nVertices % xSize){
				/* not an integer */
				Py_DECREF(vertexArray);
				Py_INCREF(Py_None);
				return(Py_None);
			}else{
				ySize = nVertices/xSize;
			}
		}else{
				Py_DECREF(vertexArray);
				Py_INCREF(Py_None);
				return(Py_None);
		}
	}

	/* I have the xSize and ySize */
	/* create the output arrays */
	dim[0] = xSize;
    xArray = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT);
    if (xArray == NULL){
        Py_DECREF(vertexArray);
	    PyErr_SetString(st->error, "Error creating x output array");
		return NULL;
    }

	dim[0] = ySize;
    yArray = (PyArrayObject *) PyArray_SimpleNew(1, dim, NPY_FLOAT);
    if (yArray == NULL){
        Py_DECREF(vertexArray);
        Py_DECREF(xArray);
	    PyErr_SetString(st->error, "Error creating y output array");
		return NULL;
    }

	/*I fill the potential values in the buffers */
	if (xRepeatedFirst == 1)
	{
		xBuffer = (float *) xArray->data;
		pv = (float *) vertexArray->data;
		for(i=0; i<xSize; i++)
		{
			*(xBuffer+i) = *(pv+(i*ySize*3));
		}

		pv++;
		yBuffer = (float *) yArray->data;
		for(i=0; i<ySize; i++)
		{
			*(yBuffer+i) = *(pv+3*i);
			/*printf("i = %d y = %f\n", i, *(yBuffer+i));*/
		}
	}else{
		yBuffer = (float *) yArray->data;
		pv = (float *) vertexArray->data;
		pv++;
		for(i=0; i<ySize; i++)
		{
			*(yBuffer+i) = *(pv+(i*xSize*3));
		}

		pv = (float *) vertexArray->data;
		xBuffer = (float *) xArray->data;
		for(i=0; i<xSize; i++)
		{
			*(xBuffer+i) = *(pv+3*i);
			//printf("i = % d x = %f y = %f\n", i, *(xBuffer+i), *(yBuffer+i));
		}
	}

	/*verify X increments follow an order */
	if(xSize > 1)
	{
		if (*xBuffer > *(xBuffer+1))
		{
			/* descending order */
		}else{
			/* ascending order */
		}
	}

	/* Verify Y increments follow an order */
	if(ySize > 1)
	{
		if (*yBuffer > *(yBuffer+1))
		{
			/* descending order */
		}else{
			/* ascending order */
		}
	}

	/* Verify if I reproduce the complete array */
	pv = (float *) vertexArray->data;
	notAMesh = 0;
	for(i=0; i<xSize; i++)
	{
		for (j=0; j<ySize; j++)
		{
			pv = (float *) PyArray_GETPTR2(vertexArray, i * ySize + j, 0);
			//printf("i=%d, j=%d, %f, %f, X = %f, Y= %f\n", i, j, *(xBuffer+i), *(yBuffer+j), *pv, *(pv+1));
			if (xRepeatedFirst)
			{
				if(fabs(*(xBuffer+i) - *pv) > delta)
				{
					//printf("X reason i = % d x = %f y = %f\n", i, *(xBuffer+i), *(yBuffer+i));
					notAMesh = 1;
					break;
				}
				pv = (float *) PyArray_GETPTR2(vertexArray, i * ySize + j, 1);
				if (fabs(*(yBuffer+j) - *pv) > delta)
				{
					printf("Y reason i = % d x = %f y = %f\n", (int) i, *(xBuffer+i), *(yBuffer+i));
					notAMesh = 1;
					break;
				}
			}else{
				if(fabs(*(xBuffer+j) - *pv) > delta)
				{
					//printf("X reason i = % d x = %f y = %f\n", i, *(xBuffer+i), *(yBuffer+i));
					notAMesh = 1;
					break;
				}
				pv = (float *) PyArray_GETPTR2(vertexArray, i * ySize + j, 1);
				if (fabs(*(yBuffer+i) - *pv) > delta)
				{
					printf("Y reason i = % d x = %f y = %f\n", (int) i, *(xBuffer+i), *(yBuffer+i));
					notAMesh = 1;
					break;
				}
			}
		}
		if (notAMesh)
		{
			break;
		}
	}
	Py_DECREF(vertexArray);
	if (notAMesh)
	{
		Py_DECREF(xArray);
		Py_DECREF(yArray);
		Py_INCREF(Py_None);
		return(Py_None);
	}
	return(Py_BuildValue("NN", PyArray_Return(xArray), PyArray_Return(yArray)));
}


static PyObject *draw3DGridTexture(PyObject *dummy, PyObject *args)
{
  // Build 3D texture
  /*
  if( _3Dbuff ) {
    glEnable(GL_TEXTURE_3D);
    glGenTextures(1, &_3Dtex);
    glBindTexture(GL_TEXTURE_3D, _3Dtex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexImage3D(GL_TEXTURE_3D, 0, 2, 256, 256, 256, 0, GL_LUMINANCE_ALPHA,
                 GL_UNSIGNED_BYTE, _3Dbuff);
    glDisable(GL_TEXTURE_3D);
  }
  */
        printf("Not implementedi yet\n");
	Py_INCREF(Py_None);
	return(Py_None);
}

static PyObject *testOpenGL(PyObject *self, PyObject *args)
{
	/*
	class a {
		int i;
	};
	a *p = new a();
	delete p;
	*/
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(-100, -100, 0);
	glVertex3f(0, 100, 0);
	glVertex3f(100, -100, 0);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *gridMarchingCubes(PyObject *self, PyObject *args)
{
	/* input parameters */
	/* x, y, z 1D arrays defining the grid */
	PyObject	   *xinput, *yinput, *zinput;
	/* array containing the values of all the points of the grid */
	PyObject	   *vinput;
	/* the isosurface value */
	float		   isoValue;
	/* the color to be used - 4 bytes - */
	PyObject		*cinput=NULL;
	/* the grid step */
	int			steps[3] = {1, 1, 1};
	/* a debug flag */
	int				 debugFlag = 0;

	/* called functions */
	extern void vSetVerticesPointer(float *);
	extern void vSetGridPointers(float *, float *, float *);
	extern void vSetValuesPointer(float *);
	extern void vSetIsoValue(float );
	extern void vSetDataSizes(int , int , int);
	extern void vSetColor(float , float , float, float);
	extern void vSetStepIncrements(int , int , int);
	extern void vMarchingCubes(void);

	/* local variables */
	PyArrayObject *xArray, *yArray, *zArray, *valuesArray;
	PyArrayObject *colorArray = NULL;
	float		color[4] = {-1.0, -1.0, -1.0, 1.0};
	int			xSize, ySize, zSize, vSize;
	unsigned char *pc;
	npy_intp	i;
    struct module_state *st = GETSTATE(self);


	/* statements */
	if (!PyArg_ParseTuple(args, "OOOOf|O(iii)i", &xinput, &yinput, &zinput, &vinput, &isoValue,\
												&cinput, &steps[0], &steps[1], &steps[2], &debugFlag))
	{
	    PyErr_SetString(st->error, "Unable to parse arguments. At least four float arrays and one float.");
        return NULL;
	}
	/* check we are not going to fall in an endless loop */
	if ((steps[0] <= 0) || (steps[1] <= 0)|| (steps[2] <= 0))
	{
		PyErr_SetString(st->error, "0 Step increment");
		return NULL;
	}
	/* convert to a contiguous array of at least 1 dimension */
	xArray = (PyArrayObject *)
    				PyArray_FROMANY(xinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (xArray == NULL)
	{
	    PyErr_SetString(st->error, "First argument cannot be converted to a float array.");
        return NULL;
	}

	/* convert to a contiguous array of at least 1 dimension */
	yArray = (PyArrayObject *)
    				PyArray_FROMANY(yinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
    if (yArray == NULL)
	{
		Py_DECREF(xArray);
	    PyErr_SetString(st->error, "Second argument cannot be converted to a float array.");
        return NULL;
	}

	/* convert to a contiguous array of at least 1 dimension */
	zArray = (PyArrayObject *)
    				PyArray_FROMANY(zinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);

    if (zArray == NULL)
	{
		Py_DECREF(xArray);
		Py_DECREF(yArray);
	    PyErr_SetString(st->error, "Third argument cannot be converted to a float array.");
        return NULL;
	}

	/* obtain the size of the arrays */
	xSize = 1;
	for (i=0; i< xArray->nd;i++){
		xSize *= xArray->dimensions[i];
	}

	ySize = 1;
	for (i=0; i< yArray->nd;i++){
		ySize *= yArray->dimensions[i];
	}

	zSize = 1;
	for (i=0; i< zArray->nd;i++){
		zSize *= zArray->dimensions[i];
	}

	/* convert to a contiguous array of at least 1 dimension */
	valuesArray = (PyArrayObject *)
				PyArray_FROMANY(vinput, NPY_FLOAT, 1, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
	if (valuesArray == NULL)
	{
		Py_DECREF(xArray);
		Py_DECREF(yArray);
		Py_DECREF(zArray);
		PyErr_SetString(st->error, "Fourth argument cannot be converted to a float array.");
		return NULL;
	}
	/* check the values array size */
	vSize = 1;
	for (i=0; i< (valuesArray)->nd;i++){
		vSize *= (valuesArray)->dimensions[i];
	}
	printf("xSize = %d, ySize = %d, zSize = %d, vSize = %d\n", xSize, ySize, zSize, vSize);

	if (vSize != (xSize) * (ySize) *  (zSize)){
		Py_DECREF(xArray);
		Py_DECREF(yArray);
		Py_DECREF(zArray);
		Py_DECREF(valuesArray);
		PyErr_SetString(st->error, "Number of values does not match number of vertices.");
		return NULL;
	}

	/* parse the (optional) color to be used */
	/* The optional input color */
	/* convert to a contiguous array of at least 1 dimension */
	if ((cinput != NULL) && (cinput != Py_None)){
		colorArray = (PyArrayObject *)
    					PyArray_ContiguousFromAny(cinput, NPY_UBYTE, 1, 0);
		if (!colorArray)
		{
			Py_DECREF(xArray);
			Py_DECREF(yArray);
			Py_DECREF(zArray);
			Py_DECREF(valuesArray);
			PyErr_SetString(st->error, "Fourth argument cannot be converted to an unsigned byte array.");
			return NULL;
		}
		pc = (unsigned char *) colorArray->data;
		color[0] = (float) ( (*pc)/255.);
		color[1] = (float) (*(pc+1)/255.);
		color[2] = (float) (*(pc+2)/255.);
		if(colorArray->dimensions[0] >3)
						color[3] = (float) (*(pc+3)/255.);
	}

	/* call the marching cubes function */
	if(debugFlag)
	{
		printf("Isosurface value = %f\n", isoValue);
		printf("Isosurface color = (%f, %f, %f, %f)\n", \
			color[0], color[1], color[2], color[3]);
		printf("Step increments  = (%d, %d, %d)\n", \
				steps[0], steps[1], steps[2]);

	}
	vSetGridPointers((float *) xArray->data, (float *) yArray->data, (float *) zArray->data);
	vSetValuesPointer((float *) valuesArray->data);
	vSetIsoValue(isoValue);
	vSetDataSizes(xSize, ySize, zSize);
	if (colorArray != NULL)
	{
		vSetColor(color[0], color[1], color[2], color[3]);
	}else{
		vSetColor(color[0], color[1], color[2], color[3]);
	}
	vSetStepIncrements(steps[0], steps[1], steps[2]);
	printf("CALLING MARCHING CUBES\n");
	vMarchingCubes();
	printf("BACK FROM MARCHING CUBES\n");

	/* clean up everything */
	Py_DECREF(xArray);
	Py_DECREF(yArray);
	Py_DECREF(zArray);
	Py_DECREF(valuesArray);
	if (colorArray != NULL)
	{
		Py_DECREF(colorArray);
	}
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *marchingCubesXYZ(PyObject *self, PyObject *args)
{
	/* input parameters  */
	PyObject	   *input1, *input2;
	int			   xSize, ySize, zSize;
	float		   isoValue;

	/*optional input parameters */
	PyObject		*inputColor = NULL;
	int				 debugFlag = 0;

	PyArrayObject *verticesArray;
	PyArrayObject *valuesArray;
	PyArrayObject *colorArray = NULL;
	int			steps[3] = {1, 1, 1};
	float		color[4] = {-1.0, -1.0, -1.0, 1.0};
	float		*p;
    struct module_state *st = GETSTATE(self);

	/* called functions */
	extern void vSetVerticesPointer(float *);
	extern void vSetValuesPointer(float *);
	extern void vSetIsoValue(float );
	extern void vSetDataSizes(int , int , int);
	extern void vSetColor(float , float , float, float);
	extern void vSetStepIncrements(int , int , int);
	extern void vMarchingCubes(void);

    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "OOiiif|O(iii)i", &input1, &input2, &xSize, &ySize, &zSize, &isoValue, \
										  &inputColor, \
										  &steps[0], &steps[1], &steps[2], &debugFlag)){
	    PyErr_SetString(st->error, "Unable to parse arguments");
        return NULL;
	}

	/* The array containing the (x, y, z) vertices */
	verticesArray = (PyArrayObject *)
    				PyArray_ContiguousFromObject(input1, PyArray_FLOAT,2,2);
    if (verticesArray == NULL){
	    PyErr_SetString(st->error, "First argument is not a nrows x 3 array");
        return NULL;
	}else{
		if(verticesArray->dimensions[1] != 3)
		{
			Py_DECREF(verticesArray);
		    PyErr_SetString(st->error, "First argument is not a nrows x 3 array");
		    return NULL;
		}
	}

	/* The array containing I(x, y, z) values */
	valuesArray = (PyArrayObject *)
    				PyArray_ContiguousFromObject(input2, PyArray_FLOAT,0,0);
    if (valuesArray == NULL){
		Py_DECREF(verticesArray);
	    PyErr_SetString(st->error, "Second argument is not a nrows x 1 array");
        return NULL;
	}

	/* The optional input color */
	if (inputColor != NULL){
		colorArray = (PyArrayObject *)
    				 PyArray_FROMANY(inputColor, NPY_FLOAT, 0, 0, NPY_C_CONTIGUOUS|NPY_FORCECAST);
		if (!colorArray){
			Py_DECREF(verticesArray);
			Py_DECREF(valuesArray);
			PyErr_SetString(st->error, "Input color is not a vector");
		    return NULL;
		}
		if(colorArray->dimensions[0] >= 3){
			p = (float *) colorArray->data;
			color[0] = *p;
			color[1] = *(p+1);
			color[2] = *(p+2);
			if(colorArray->dimensions[0] >3)
							color[3] = *(p+3);
		}
	}

	/* for the time being I assume the dimensions are correct
	   this means verticesArray is an nrows x 3 array and
	   valuesArray is an nrow x 1 array. nrows is equal to the
	   product xSize, ySize, zSize */

	if(debugFlag)
	{
		printf("Isosurface value = %f\n", isoValue);
		printf("Isosurface color = (%f, %f, %f, %f)\n", \
			color[0], color[1], color[2], color[3]);
		printf("Step increments  = (%d, %d, %d)\n", \
				steps[0], steps[1], steps[2]);

	}

	vSetVerticesPointer((float *) verticesArray->data);
	vSetValuesPointer((float *) valuesArray->data);
	vSetIsoValue(isoValue);
	vSetDataSizes(xSize, ySize, zSize);
	if (inputColor != NULL)
		vSetColor(color[0], color[1], color[2], color[3]);
	if ((steps[0] == 0) || (steps[1] == 0)|| (steps[2] == 0))
	{
		Py_DECREF(verticesArray);
		Py_DECREF(valuesArray);
		if (colorArray != NULL)
		{
			Py_DECREF(colorArray);
		}
		PyErr_SetString(st->error, "0 Step increment");
		return NULL;
	}
	vSetStepIncrements(steps[0], steps[1], steps[2]);
	vMarchingCubes();

	Py_DECREF(verticesArray);
	Py_DECREF(valuesArray);
	if (colorArray != NULL)
	{
		Py_DECREF(colorArray);
	}
	Py_INCREF(Py_None);
	return Py_None;

}

static PyObject *getGridFacetsFromVertices(PyObject *self, PyObject *args)
{
	/* input parameters  */
	PyObject *input1;
	int	xsize, ysize;

	PyArrayObject *inputArray;
	PyArrayObject *result;
	int inputArrayDimensions[2];
	int outputArrayDimensions[2];
	int i, j, index;
	float  *resultP;
    struct module_state *st = GETSTATE(self);

    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "Oii", &input1, &xsize, &ysize))
        return NULL;

	inputArray = (PyArrayObject *)
				PyArray_ContiguousFromObject(input1, PyArray_FLOAT,0,0);


    if (inputArray == NULL)
	{
        return NULL;
	}
	if (inputArray->nd != 2)
	{
		PyErr_SetString(st->error,
				"Expected a nrows x three columns array as input");
		Py_DECREF(inputArray);
	}

	inputArrayDimensions[0] = inputArray->dimensions[0];
	inputArrayDimensions[1] = inputArray->dimensions[1];

	if ((inputArrayDimensions[0] <= 1) || (inputArrayDimensions[1] != 3))
	{
		PyErr_SetString(st->error,
				"Expected a nrows (>1) x three columns array as input");
		Py_DECREF(inputArray);
	}

	/* input checked */

	/* create output array */
    outputArrayDimensions[0] = 3 * xsize * ysize *2;
	outputArrayDimensions[1] = 3;
	result = (PyArrayObject *)
				PyArray_FromDims(2, outputArrayDimensions, PyArray_FLOAT);
    if (result == NULL){
        Py_DECREF(inputArray);
        return NULL;
    }

	resultP = (float *) result->data;

	for (i=0; i< (xsize-1); i++)
	{
		for (j=0; i< (ysize-1); j++)
		{
			/* first triangle */
			/* side 1*/
			index   = i*ysize +j;
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;

			/* side 2 */
			index = (i+1)*ysize +j;
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;

			/* side 3 */
			index = i*ysize +(j+1);
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;

			/* second triangle */
			/* side 1*/
			index   = (i+1)*ysize +j;
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;

			/* side 2 */
			index = (i+1)*ysize +(j+1);
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;

			/* side 3 */
			index = i*ysize +(j+1);
			*resultP = inputArray->data[index];
			resultP++;
			*resultP = inputArray->data[index+1];
			resultP++;
			*resultP = inputArray->data[index+2];
			resultP++;
		}
	}
	Py_DECREF(inputArray);
	return PyArray_Return(result);
}


/* Tell Python wich methods are available in this module. */
static PyMethodDef Object3DCToolsMethods[] = {
	{"get2DGridFromXY",  get2DGridFromXY,  METH_VARARGS},
	{"draw2DGridPoints", draw2DGridPoints, METH_VARARGS},
	{"draw2DGridLines",  draw2DGridLines,  METH_VARARGS},
	{"draw2DGridQuads",  draw2DGridQuads,  METH_VARARGS},
	{"get3DGridFromXYZ", get3DGridFromXYZ, METH_VARARGS},
	{"draw3DGridPoints", draw3DGridPoints, METH_VARARGS},
	{"draw3DGridLines",  draw3DGridLines,  METH_VARARGS},
	{"draw3DGridQuads",  draw3DGridQuads,  METH_VARARGS},
	{"draw3DGridTexture",  draw3DGridTexture,  METH_VARARGS},
	{"drawXYZPoints",    drawXYZPoints,	   METH_VARARGS},
	{"drawXYZLines",     drawXYZLines,     METH_VARARGS},
	{"drawXYZTriangles", drawXYZTriangles, METH_VARARGS},
	{"getVertexArrayMeshAxes", getVertexArrayMeshAxes, METH_VARARGS},
	{"gridMarchingCubes",gridMarchingCubes, METH_VARARGS},
	{"marchingCubesXYZ", marchingCubesXYZ,  METH_VARARGS},
	{"getGridFacetsFromVertices", getGridFacetsFromVertices, METH_VARARGS},
	{"testOpenGL", testOpenGL, METH_VARARGS},
	{NULL, NULL, 0, NULL} /* sentinel */
};


#if PY_MAJOR_VERSION >= 3

static int Object3DCTools_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int Object3DCTools_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "Object3DCTools",     /* m_name */
    "This is a module",   /* m_doc */
    sizeof(struct module_state), /* m_size */
    Object3DCToolsMethods,   /* m_methods */
    NULL,                    /* m_reload */
    Object3DCTools_traverse, /* m_traverse */
    Object3DCTools_clear,    /* m_clear */
    NULL,                    /* m_free */
};

#define INITERROR return NULL

PyObject *
PyInit_Object3DCTools(void)

#else
#define INITERROR return

void
initObject3DCTools(void)
#endif
{
    struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("Object3DCTools", Object3DCToolsMethods);
#endif

    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("Object3DCTools.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array()

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
