#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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
#include <Python.h>
/* adding next line may raise errors
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
*/
#include <./numpy/arrayobject.h>
#include <math.h>

#ifndef NPY_ARRAY_ENSURECOPY
#define NPY_ARRAY_ENSURECOPY NPY_ENSURECOPY
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
#define isARRAY(a) ((a) && PyArray_Check((PyArrayObject *)a))
#define A_SIZE(a) PyArray_Size((PyObject *) a)
#define isARRAY(a) ((a) && PyArray_Check((PyArrayObject *)a))
#ifndef WIN32
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#else
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define M_PI 3.1415926535
#define erf myerf
#define erfc myerfc
#endif
#define MAX_SAVITSKY_GOLAY_WIDTH 101
#define MIN_SAVITSKY_GOLAY_WIDTH 3

/* SNIP related functions */
void lls(double *data, int size);
void lls_inv(double *data, int size);
void snip1d(double *data, int size, int width);
void snip1d_multiple(double *data, int n_channels, int snip_width, int n_spectra);
void snip2d(double *data, int nrows, int ncolumns, int width);
void snip3d(double *data, int nx, int ny, int nz, int width);
void lsdf(double *data, int size, int fwhm, double f, double A, double M, double ratio);
void smooth1d(double *data, int size);
void smooth2d(double *data, int size0, int size1);
void smooth3d(double *data, int size0, int size1, int size2);
/* end of SNIP related functions */

/* --------------------------------------------------------------------- */

static PyObject *
SpecfitFuns_snip1d(PyObject *self, PyObject *args)
{
    PyObject *input;
    double width0 = 50.;
    int smooth_iterations = 0;
    int llsflag = 0;
    PyArrayObject   *ret;
    double *doublePointer;
    int i, n, n_channels, n_spectra, width;

    if (!PyArg_ParseTuple(args, "Od|ii", &input, &width0, &smooth_iterations, &llsflag))
        return NULL;

    ret = (PyArrayObject *)
             PyArray_FROMANY(input, NPY_DOUBLE, 1, 2, NPY_ARRAY_ENSURECOPY);

    if (ret == NULL){
        printf("Cannot create 1D array from input\n");
        return NULL;
    }

    if(PyArray_NDIM(ret) == 1)
    {
        n_spectra = 1;
        n_channels = (int) (PyArray_DIMS(ret)[0]);
    }
    else
    {
        n_spectra = (int) (PyArray_DIMS(ret)[0]);
        n_channels = (int) (PyArray_DIMS(ret)[1]);
    }

    width = (int )width0;

    for (n = 0; n < n_spectra; n++)
    {
        for (i=0; i<smooth_iterations; i++)
        {
            doublePointer = (double *) PyArray_DATA(ret);
            smooth1d(&(doublePointer[n*n_channels]), n_channels);
        }
        if (llsflag)
        {
            doublePointer = (double *) PyArray_DATA(ret);
            lls(&(doublePointer[n*n_channels]), n_channels);
        }
    }


    snip1d_multiple((double *) PyArray_DATA(ret), n_channels, width, n_spectra);

    for (n = 0; n < n_spectra; n++)
    {
        if (llsflag)
        {
            doublePointer = (double *) PyArray_DATA(ret);
            lls_inv(&(doublePointer[n*n_channels]), n_channels);
        }
    }

    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_snip2d(PyObject *self, PyObject *args)
{
    PyObject *input;
    double width0 = 50.;
    int smooth_iterations = 0;
    int llsflag = 0;
    PyArrayObject   *ret;
    int i, nrows, ncolumns, size, width;

    if (!PyArg_ParseTuple(args, "Od|ii", &input, &width0, &smooth_iterations, &llsflag))
        return NULL;

    ret = (PyArrayObject *)
             PyArray_FROMANY(input, NPY_DOUBLE, 2, 2, NPY_ARRAY_ENSURECOPY);

    if (ret == NULL){
        printf("Cannot create 2D array from input\n");
        return NULL;
    }

    size = 1;
    for (i=0; i<PyArray_NDIM(ret); i++)
    {
        size = (int) (size * PyArray_DIMS(ret)[i]);
    }
    nrows = (int) PyArray_DIMS(ret)[0];
    ncolumns = (int) PyArray_DIMS(ret)[1];

    width = (int )width0;

    for (i=0; i<smooth_iterations; i++)
    {
        smooth2d((double *) PyArray_DATA(ret), nrows, ncolumns);
    }

    if (llsflag)
    {
        lls((double *) PyArray_DATA(ret), size);
    }

    snip2d((double *) PyArray_DATA(ret), nrows, ncolumns, width);

    if (llsflag)
    {
        lls_inv((double *) PyArray_DATA(ret), size);
    }

    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_snip3d(PyObject *self, PyObject *args)
{
    PyObject *input;
    double width0 = 50.;
    int smooth_iterations = 0;
    int llsflag = 0;
    PyArrayObject   *ret;
    int i, nx, ny, nz, size, width;

    if (!PyArg_ParseTuple(args, "Od|ii", &input, &width0, &smooth_iterations, &llsflag))
        return NULL;

    ret = (PyArrayObject *)
             PyArray_FROMANY(input, NPY_DOUBLE, 3, 3, NPY_ARRAY_ENSURECOPY);

    if (ret == NULL){
        printf("Cannot create 3D array from input\n");
        return NULL;
    }

    size = 1;
    for (i=0; i<PyArray_NDIM(ret); i++)
    {
        size = (int) (size * PyArray_DIMS(ret)[i]);
    }
    nx = (int) PyArray_DIMS(ret)[0];
    ny = (int) PyArray_DIMS(ret)[1];
    nz = (int) PyArray_DIMS(ret)[2];

    width = (int )width0;

    for (i=0; i<smooth_iterations; i++)
    {
        smooth3d((double *) PyArray_DATA(ret), nx, ny, nz);
    }

    if (llsflag)
    {
        lls((double *) PyArray_DATA(ret), size);
    }

    snip3d((double *) PyArray_DATA(ret), nx, ny, nz, width);

    if (llsflag)
    {
        lls_inv((double *) PyArray_DATA(ret), size);
    }

    return PyArray_Return(ret);
}

/* end SNIP algorithm */

/* Function SUBAC returning smoothed array */

static PyObject *
SpecfitFuns_subacold(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *iarray, *ret;
    npy_intp n, dimensions[1];
    double niter0 = 5000.;
    int i, j, niter = 5000;
    double  t_old, t_mean, c = 1.0000;
    double  *data;

    if (!PyArg_ParseTuple(args, "O|dd", &input, &c, &niter0))
        return NULL;
    iarray = (PyArrayObject *)
             PyArray_CopyFromObject(input, NPY_DOUBLE,1,1);
    if (iarray == NULL)
        return NULL;
    niter = (int ) niter0;
    n = PyArray_DIMS(iarray)[0];
    dimensions[0] = PyArray_DIMS(iarray)[0];
    ret = (PyArrayObject *) PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(iarray);
        return NULL;
    }

    /* Do the job */
    data = (double *) PyArray_DATA(iarray);
    for (i=0;i<niter;i++){
        t_old = *(data);
        for (j=1;j<n-1;j++) {
            t_mean = 0.5 * (t_old + *(data+j+1));
            t_old = *(data+j);
            if (t_old > (t_mean * c))
                *(data+j) = t_mean;
       }
    /*   t_mean = 0.5 * (t_old + *(data+n-1));
       t_old = *(data+n-1);
       if (t_old > (t_mean * c))
                *(data+n-1) = t_mean;*/
    }
    ret = (PyArrayObject *) PyArray_Copy(iarray);
    Py_DECREF(iarray);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);

}

static PyObject *
SpecfitFuns_subac(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *iarray, *ret, *anchors;
    int n;
    npy_intp dimensions[1];
    double niter0 = 5000.;
    double deltai0= 1;
    PyObject *anchors0 = NULL;
    int i, j, k, l, deltai = 1,niter = 5000;
    double  t_mean, c = 1.000;
    double  *data, *retdata;
    int     *anchordata;
    int nanchors, notdoit;
    int    notdone=1;

    if (!PyArg_ParseTuple(args, "O|dddO", &input, &c, &niter0,&deltai0, &anchors0))
        return NULL;
    iarray = (PyArrayObject *)
             PyArray_CopyFromObject(input, NPY_DOUBLE,1,1);
    if (iarray == NULL)
        return NULL;
    deltai= (int ) deltai0;
    if (deltai <=0) deltai = 1;
    niter = (int ) niter0;
    n = (int) PyArray_DIMS(iarray)[0];
    dimensions[0] = PyArray_DIMS(iarray)[0];
    ret = (PyArrayObject *) PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(iarray);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);
    memcpy(PyArray_DATA(ret), PyArray_DATA(iarray), PyArray_DIMS(iarray)[0] * sizeof(double));

    if (n < (2*deltai+1)){
        /*ret = (PyArrayObject *) PyArray_Copy(array);*/
        Py_DECREF(iarray);
        return PyArray_Return(ret);
    }
    /* do the job */
    data   = (double *) PyArray_DATA(iarray);
    retdata   = (double *) PyArray_DATA(ret);

    if (anchors0 != NULL)
    {
        if (PySequence_Check(anchors0)){
            anchors = (PyArrayObject *)
                 PyArray_ContiguousFromObject(anchors0, NPY_INT, 1, 1);
            if (anchors == NULL)
            {
                Py_DECREF(iarray);
                Py_DECREF(ret);
                return NULL;
            }
            anchordata = (int *) PyArray_DATA(anchors);
            nanchors   = (int) PySequence_Size(anchors0);
            for (i=0;i<niter;i++){
                for (j=deltai;j<n-deltai;j++) {
                    notdoit = 0;
                    for (k=0; k<nanchors; k++)
                    {
                        l =*(anchordata+k);
                        if (j>(l-deltai))
                        {
                            if (j<(l+deltai))
                            {
                                notdoit = 1;
                                break;
                            }
                        }
                    }
                    if (notdoit)
                        continue;
                    t_mean = 0.5 * (*(data+j-deltai) + *(data+j+deltai));
                    if (*(retdata+j) > (t_mean * c))
                                *(retdata+j) = t_mean;
                }
                memcpy(PyArray_DATA(iarray), PyArray_DATA(ret), PyArray_DIMS(iarray)[0] * sizeof(double));
            }
            Py_DECREF(anchors);
            notdone = 0;
        }
    }
    if (notdone)
    {
        for (i=0;i<niter;i++){
            for (j=deltai;j<n-deltai;j++) {
                t_mean = 0.5 * (*(data+j-deltai) + *(data+j+deltai));
            if (*(retdata+j) > (t_mean * c))
                    *(retdata+j) = t_mean;
            }
            memcpy(PyArray_DATA(iarray), PyArray_DATA(ret), PyArray_DIMS(iarray)[0] * sizeof(double));
        }
    }
    Py_DECREF(iarray);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);

}

static PyObject *
SpecfitFuns_subacfast(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *iarray, *ret, *anchors;
    npy_intp n, dimensions[1];
    double niter0 = 5000.;
    double deltai0= 1;
    PyObject *anchors0 = NULL;
    int i, j, k, l, deltai = 1,niter = 5000;
    double  t_mean, c = 1.000;
    double  *retdata;
    int     *anchordata;
    int nanchors, notdoit;

    if (!PyArg_ParseTuple(args, "O|dddO", &input, &c, &niter0,&deltai0, &anchors0))
        return NULL;
    iarray = (PyArrayObject *)
             PyArray_CopyFromObject(input, NPY_DOUBLE,1,1);
    if (iarray == NULL)
        return NULL;
    deltai= (int ) deltai0;
    if (deltai <=0) deltai = 1;
    niter = (int ) niter0;
    n = PyArray_DIMS(iarray)[0];
    dimensions[0] = PyArray_DIMS(iarray)[0];
    ret = (PyArrayObject *) PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(iarray);
        return NULL;
    }
    memcpy(PyArray_DATA(ret), PyArray_DATA(iarray), PyArray_DIMS(iarray)[0] * sizeof(double));

    if (n < (2*deltai+1)){
        /*ret = (PyArrayObject *) PyArray_Copy(array);*/
        Py_DECREF(iarray);
        return PyArray_Return(ret);
    }
    /* do the job */
    retdata   = (double *) PyArray_DATA(ret);
    if (PySequence_Check(anchors0)){
        anchors = (PyArrayObject *)
             PyArray_ContiguousFromObject(anchors0, NPY_INT, 1, 1);
        if (anchors == NULL)
        {
            Py_DECREF(iarray);
            Py_DECREF(ret);
            return NULL;
        }
        anchordata = (int *) PyArray_DATA(anchors);
        nanchors   = (int) PySequence_Size(anchors0);
        memcpy(PyArray_DATA(iarray), PyArray_DATA(ret), PyArray_DIMS(iarray)[0] * sizeof(double));
        for (i=0;i<niter;i++){
            for (j=deltai;j<n-deltai;j++) {
                notdoit = 0;
                for (k=0; k<nanchors; k++)
                {
                    l =*(anchordata+k);
                    if (j>(l-deltai))
                    {
                        if (j<(l+deltai))
                        {
                            notdoit = 1;
                            break;
                        }
                    }
                }
                if (notdoit)
                continue;
                t_mean = 0.5 * (*(retdata+j-deltai) + *(retdata+j+deltai));
                if (*(retdata+j) > (t_mean * c))
                        *(retdata+j) = t_mean;
            }
        }
        Py_DECREF(anchors);
    }
    else
    {
        memcpy(PyArray_DATA(iarray), PyArray_DATA(ret), PyArray_DIMS(iarray)[0] * sizeof(double));
        for (i=0;i<niter;i++){
            for (j=deltai;j<n-deltai;j++) {
                t_mean = 0.5 * (*(retdata+j-deltai) + *(retdata+j+deltai));
            if (*(retdata+j) > (t_mean * c))
                    *(retdata+j) = t_mean;
            }
        }
    }
    Py_DECREF(iarray);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);

}

static PyObject *
SpecfitFuns_gauss(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2;
    double  *px, *pret;
    const char *tpe;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm;
    } gaussian;
    gaussian *pgauss;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    if (debug == 1){
        tpe = input1->ob_type->tp_name;
            printf("C(iotest): input1 type of object = %s\n",tpe);
       /* j = PyObject_Length (input1);
        printf("Length = %d\n",j);
        for (i=0;i<j;i++){

            printf("Element %d = %ld\n",i,
                            PyInt_AsLong(PyList_GetItem(input1,i)));
        }
        */

    }

    param = (PyArrayObject *)
             PyArray_ContiguousFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            dhelp = pgauss[i].fwhm/(2.0*sqrt(2.0*log2));
            dhelp = (*px - pgauss[i].centroid)/dhelp;
            if (dhelp <= 20) {
                *pret += pgauss[i].height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            pgauss = (gaussian *) PyArray_DATA(param);
            for (i=0;i<(npars/3);i++){
                dhelp = pgauss[i].fwhm/(2.0*sqrt(2.0*log2));
                dhelp = (*px - pgauss[i].centroid)/dhelp;
                if (dhelp <= 20) {
                *pret += pgauss[i].height * exp (-0.5 * dhelp * dhelp);
                }
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_agauss(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, dhelp0,log2, sqrt2PI,sigma,tosigma;
    double  *px, *pret;
    typedef struct {
        double  area;
        double  centroid;
        double  fwhm;
    } gaussian;
    gaussian *pgauss;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            sigma = pgauss[i].fwhm*tosigma;
            dhelp = (*px - pgauss[i].centroid)/sigma;
            if (dhelp <= 35){
                *pret += (pgauss[i].area/(sigma*sqrt2PI))* exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            sigma  = pgauss[i].fwhm*tosigma;
            dhelp0 = pgauss[i].area/(sigma*sqrt2PI);
            px = (double *) PyArray_DATA(x);
            pret = (double *) PyArray_DATA(ret);
            for (j=0;j<k;j++){
                if (i==0)
                    *pret = 0.0;
                dhelp = (*px - pgauss[i].centroid)/sigma;
                if (dhelp <= 35){
                    *pret += dhelp0 * exp (-0.5 * dhelp * dhelp);
                }
                pret++;
                px++;
            }
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}
static PyObject *
SpecfitFuns_fastagauss(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k,expindex;
    double  dhelp, dhelp0,log2, sqrt2PI,sigma,tosigma;
    double  *px, *pret;
    static double EXP[5000];
    typedef struct {
        double  area;
        double  centroid;
        double  fwhm;
    } gaussian;
    gaussian *pgauss;

    /* initialisation */
    if (EXP[0] < 1){
        for (i=0;i<5000;i++){
            EXP[i] = exp(-0.01 * i);
        }
    }

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            sigma = pgauss[i].fwhm*tosigma;
            dhelp = (*px - pgauss[i].centroid)/sigma;
            if (dhelp <= 35){
                *pret += (pgauss[i].area/(sigma*sqrt2PI))* exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            sigma  = pgauss[i].fwhm*tosigma;
            dhelp0 = pgauss[i].area/(sigma*sqrt2PI);
            px = (double *) PyArray_DATA(x);
            pret = (double *) PyArray_DATA(ret);
            for (j=0;j<k;j++){
                if (i==0)
                    *pret = 0.0;
                dhelp = (*px - pgauss[i].centroid)/sigma;
                if (dhelp <= 15){
                    dhelp = 0.5 * dhelp * dhelp;
                    if (dhelp < 50){
                        expindex = (int) (dhelp * 100);
                        *pret += dhelp0 * EXP[expindex]*(1.0 - (dhelp - 0.01 * expindex)) ;
                    }else if (dhelp < 100) {
                        expindex = (int) (dhelp * 10);
              *pret += dhelp0 * pow(EXP[expindex]*(1.0 - (dhelp - 0.1 * expindex)),10) ;
                    }else if (dhelp < 1000){
                        expindex = (int) (dhelp);
             *pret += dhelp0 * pow(EXP[expindex]*(1.0 - (dhelp - expindex)),20) ;
                    }
                }
                pret++;
                px++;
            }
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_splitgauss(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2;
    double  *px, *pret;
    const char *tpe;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm1;
        double  fwhm2;
    } gaussian;
    gaussian *pgauss;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    if (debug == 1){
        tpe = input1->ob_type->tp_name;
            printf("C(iotest): input1 type of object = %s\n",tpe);
    }

    param = (PyArrayObject *)
             PyArray_ContiguousFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d rows and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d rows and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = (*px - pgauss[i].centroid) * (2.0*sqrt(2.0*log2));
            if (dhelp > 0)
            {
                dhelp = dhelp/pgauss[i].fwhm2;
            }else{
                dhelp = dhelp/pgauss[i].fwhm1;
            }
            if (dhelp <= 20) {
                *pret += pgauss[i].height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            pgauss = (gaussian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
                dhelp = (*px - pgauss[i].centroid) * (2.0*sqrt(2.0*log2));
                if (dhelp > 0)
                {
                    dhelp = dhelp /pgauss[i].fwhm2;
                }else{
                    dhelp = dhelp /pgauss[i].fwhm1;
                }
                if (dhelp <= 20) {
                    *pret += pgauss[i].height * exp (-0.5 * dhelp * dhelp);
                }
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_apvoigt(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2, sqrt2PI,sigma,tosigma;
    double  *px, *pret;
    typedef struct {
        double  area;
        double  centroid;
        double  fwhm;
        double  eta;
    } pvoigtian;
    pvoigtian *ppvoigt;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * \
                (ppvoigt[i].area / (0.5 * M_PI * ppvoigt[i].fwhm * dhelp));
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * \
                (ppvoigt[i].area / (0.5 * M_PI * ppvoigt[i].fwhm * dhelp));
            }
            pret++;
            px++;
        }
    }

    /* The lorentzian term is calculated */
    /* Now it has to calculate the gaussian term */
    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            sigma = ppvoigt[i].fwhm * tosigma;
            dhelp = (*px - ppvoigt[i].centroid)/sigma;
            if (dhelp <= 35) {
                *pret += (1.0 - ppvoigt[i].eta) * \
                        (ppvoigt[i].area/(sigma*sqrt2PI)) \
                        * exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
                sigma = ppvoigt[i].fwhm * tosigma;
                dhelp = (*px - ppvoigt[i].centroid)/sigma;
                if (dhelp <= 35) {
                    *pret += (1.0 - ppvoigt[i].eta) * \
                        (ppvoigt[i].area/(sigma*sqrt2PI)) \
                        * exp (-0.5 * dhelp * dhelp);
                }
        }
            pret++;
            px++;
        }
    }



    /* word done */
    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}



static PyObject *
SpecfitFuns_pvoigt(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm;
        double  eta;
    } pvoigtian;
    pvoigtian *ppvoigt;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * (ppvoigt[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * (ppvoigt[i].height / dhelp);
            }
            pret++;
            px++;
        }
    }

    /* The lorentzian term is calculated */
    /* Now it has to calculate the gaussian term */
    log2 = 0.69314718055994529;

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = ppvoigt[i].fwhm/(2.0*sqrt(2.0*log2));
            dhelp = (*px - ppvoigt[i].centroid)/dhelp;
            if (dhelp <= 35) {
                *pret += (1.0 - ppvoigt[i].eta) * ppvoigt[i].height \
                        * exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
                dhelp = ppvoigt[i].fwhm/(2.0*sqrt(2.0*log2));
                dhelp = (*px - ppvoigt[i].centroid)/dhelp;
                if (dhelp <= 35) {
                *pret += (1.0 - ppvoigt[i].eta) * ppvoigt[i].height \
                        * exp (-0.5 * dhelp * dhelp);
                }
        }
            pret++;
            px++;
        }
    }



    /* word done */
    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_splitpvoigt(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm1;
        double  fwhm2;
        double  eta;
    } pvoigtian;
    pvoigtian *ppvoigt;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%5) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d rows and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d rows and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/5);i++){
            dhelp = (*px - ppvoigt[i].centroid);
            if (dhelp > 0){
                dhelp = dhelp /(0.5 * ppvoigt[i].fwhm2);
            }else{
                dhelp = dhelp /(0.5 * ppvoigt[i].fwhm1);
            }
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * (ppvoigt[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/5);i++){
            dhelp = (*px - ppvoigt[i].centroid);
            if (dhelp > 0){
                dhelp = dhelp /(0.5 * ppvoigt[i].fwhm2);
            }else{
                dhelp = dhelp /(0.5 * ppvoigt[i].fwhm1);
            }
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * (ppvoigt[i].height / dhelp);
            }
            pret++;
            px++;
        }
    }

    /* The lorentzian term is calculated */
    /* Now it has to calculate the gaussian term */
    log2 = 0.69314718055994529;

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
        ppvoigt = (pvoigtian *) PyArray_DATA(param);
        for (i=0;i<(npars/5);i++){
            dhelp = (*px - ppvoigt[i].centroid);
            if (dhelp >0){
                dhelp = dhelp /(ppvoigt[i].fwhm2/(2.0*sqrt(2.0*log2)));
            }else{
                dhelp = dhelp /(ppvoigt[i].fwhm1/(2.0*sqrt(2.0*log2)));
            }
            if (dhelp <= 35) {
                *pret += (1.0 - ppvoigt[i].eta) * ppvoigt[i].height \
                        * exp (-0.5 * dhelp * dhelp);
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            ppvoigt = (pvoigtian *) PyArray_DATA(param);
            for (i=0;i<(npars/5);i++){
                dhelp = (*px - ppvoigt[i].centroid);
                if (dhelp > 0){
                    dhelp = dhelp /(ppvoigt[i].fwhm2/(2.0*sqrt(2.0*log2)));
                }else{
                    dhelp = dhelp /(ppvoigt[i].fwhm1/(2.0*sqrt(2.0*log2)));
                }
                if (dhelp <= 35) {
                *pret += (1.0 - ppvoigt[i].eta) * ppvoigt[i].height \
                        * exp (-0.5 * dhelp * dhelp);
                }
        }
            pret++;
            px++;
        }
    }



    /* word done */
    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_lorentz(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm;
    } lorentzian;
    lorentzian *plorentz;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        plorentz = (lorentzian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += (plorentz[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            plorentz = (lorentzian *) PyArray_DATA(param);
            for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += (plorentz[i].height / dhelp);
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}


static PyObject *
SpecfitFuns_alorentz(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp;
    double  *px, *pret;
    typedef struct {
        double  area;
        double  centroid;
        double  fwhm;
    } lorentzian;
    lorentzian *plorentz;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        plorentz = (lorentzian *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += plorentz[i].area /(0.5 * M_PI * plorentz[i].fwhm * dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            plorentz = (lorentzian *) PyArray_DATA(param);
            for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += plorentz[i].area /(0.5 * M_PI * plorentz[i].fwhm * dhelp);
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_splitlorentz(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm1;
        double  fwhm2;
    } lorentzian;
    lorentzian *plorentz;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d rows and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d rows and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       *pret = 0;
        plorentz = (lorentzian *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = *px - plorentz[i].centroid;
            if (dhelp > 0){
                dhelp = dhelp /(0.5 * plorentz[i].fwhm2);
            }else{
                dhelp = dhelp /(0.5 * plorentz[i].fwhm1);
            }
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += (plorentz[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            plorentz = (lorentzian *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
            dhelp = *px - plorentz[i].centroid;
                if (dhelp > 0){
                    dhelp = dhelp /(0.5 * plorentz[i].fwhm2);
                }else{
                    dhelp = dhelp /(0.5 * plorentz[i].fwhm1);
                }
                dhelp = 1.0 + (dhelp * dhelp);
                *pret += (plorentz[i].height / dhelp);
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_downstep(PyObject *self, PyObject *args)
{
    double erfc(double);
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, tosigma;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm;
    } errorfc;
    errorfc *perrorfc;


    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));


    if (nd_x == 0){
       *pret = 0;
        perrorfc = (errorfc *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            dhelp = perrorfc[i].fwhm * tosigma;
            dhelp = (*px - perrorfc[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorfc[i].height * 0.5 * erfc(dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorfc = (errorfc *) PyArray_DATA(param);
            for (i=0;i<(npars/3);i++){
            dhelp = perrorfc[i].fwhm * tosigma;
            dhelp = (*px - perrorfc[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorfc[i].height * 0.5 * erfc(dhelp);
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_upstep(PyObject *self, PyObject *args)
{
    double erf(double);
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, tosigma;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  centroid;
        double  fwhm;
    } errorf;
    errorf *perrorf;


    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));

    if (nd_x == 0){
       *pret = 0;
        perrorf = (errorf *) PyArray_DATA(param);
        for (i=0;i<(npars/3);i++){
            dhelp = perrorf[i].fwhm * tosigma;
            dhelp = (*px - perrorf[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorf[i].height * 0.5 * (1.0 + erf(dhelp));
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorf = (errorf *) PyArray_DATA(param);
            for (i=0;i<(npars/3);i++){
            dhelp = perrorf[i].fwhm * tosigma;
            dhelp = (*px - perrorf[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorf[i].height * 0.5 * (1.0 + erf(dhelp));
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}
static PyObject *
SpecfitFuns_slit(PyObject *self, PyObject *args)
{
    double erf(double);
    double erfc(double);
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, dhelp1,dhelp2,centroid1,centroid2,tosigma;
    double  *px, *pret;
    typedef struct {
        double  height;
        double  position;
        double  fwhm;
        double  beamfwhm;
    } errorf;
    errorf *perrorf;


    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|i", &input1,&input2,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));

    if (nd_x == 0){
       *pret = 0;
        perrorf = (errorf *) PyArray_DATA(param);
        for (i=0;i<(npars/4);i++){
            dhelp = perrorf[i].beamfwhm * tosigma;
            centroid1=perrorf[i].position - 0.5 * perrorf[i].fwhm;
            centroid2=perrorf[i].position + 0.5 * perrorf[i].fwhm;
            dhelp1 = (*px - centroid1) / (sqrt(2)*dhelp);
            dhelp2 = (*px - centroid2) / (sqrt(2)*dhelp);
            *pret += perrorf[i].height * 0.5 * (1.0 + erf(dhelp1))*erfc(dhelp2);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorf = (errorf *) PyArray_DATA(param);
            for (i=0;i<(npars/4);i++){
            dhelp = perrorf[i].beamfwhm * tosigma;
            centroid1=perrorf[i].position - 0.5 * perrorf[i].fwhm;
            centroid2=perrorf[i].position + 0.5 * perrorf[i].fwhm;
            dhelp1 = (*px - centroid1) / (sqrt(2)*dhelp);
            dhelp2 = (*px - centroid2) / (sqrt(2)*dhelp);
            *pret += perrorf[i].height * 0.5 * (1.0 + erf(dhelp1))*erfc(dhelp2);
            }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_erfc(PyObject *self, PyObject *args)
{
    double erfc(double);
    PyObject *input1;
    int debug=0;
    PyArrayObject   *x;
    PyArrayObject   *ret;
    int nd_x;
    npy_intp dim_x[2];
    int j, k;
    double  dhelp;
    double  *px, *pret;


    /** statements **/
    if (!PyArg_ParseTuple(args, "O|i", &input1,&debug))
        return NULL;

    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (x == NULL){
        return NULL;
    }

    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if(debug !=0) {
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       dhelp = *px;
       *pret = erfc(dhelp);
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            dhelp = *px;
            *pret = erfc(dhelp);
            pret++;
            px++;
        }
    }

    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_erf(PyObject *self, PyObject *args)
{
    double erfc(double);
    double erf(double);
    PyObject *input1;
    int debug=0;
    PyArrayObject   *x;
    PyArrayObject   *ret;
    int nd_x;
    npy_intp dim_x[2];
    int j, k;
    double  dhelp;
    double  *px, *pret;


    /** statements **/
    if (!PyArg_ParseTuple(args, "O|i", &input1,&debug))
        return NULL;

    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (x == NULL){
        return NULL;
    }

    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    if(debug !=0) {
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if (nd_x == 0){
       dhelp = *px;
       *pret = erf(dhelp);
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            dhelp = *px;
            *pret = erf(dhelp);
            pret++;
            px++;
        }
    }

    Py_DECREF(x);
    return PyArray_Return(ret);
}

static PyObject *
SpecfitFuns_ahypermet(PyObject *self, PyObject *args)
{
    double erfc(double);
    PyObject *input1, *input2;
    int debug=0;
    int tails=15;
    int expected_pars;
    int g_term_flag, st_term_flag, lt_term_flag, step_term_flag;
    /*double g_term, st_term, lt_term, step_term;*/
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2, sqrt2PI,tosigma;
    double x1, x2, x3, x4, x5, x6, x7, x8;
    double z0, z1, z2;
    double  *px, *pret;
    typedef struct {
        double  area;
        double  position;
        double  fwhm;
        double  st_area_r;
        double  st_slope_r;
        double  lt_area_r;
        double  lt_slope_r;
        double  step_height_r;
    } hypermet;
    hypermet *phyper;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|ii", &input1,&input2,&tails,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }
    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    /* The gaussian terms must always be there */
    if(tails <= 0){
        /* I give back a matrix filled with zeros */
        ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
        if (ret == NULL){
            Py_DECREF(param);
            Py_DECREF(x);
            return NULL;
        }else{
            PyArray_FILLWBYTE(ret, 0);
            Py_DECREF(param);
            Py_DECREF(x);
            return PyArray_Return(ret);
        }
    }else{
        g_term_flag    = tails & 1;
        st_term_flag   = (tails>>1) & 1;
        lt_term_flag   = (tails>>2) & 1;
        step_term_flag = (tails>>3) & 1;
    }
    if (debug){
        printf("flags g = %d st = %d lt = %d step = %d\n",\
               g_term_flag,st_term_flag,lt_term_flag,step_term_flag);
    }
    expected_pars = 3 + st_term_flag * 2+lt_term_flag * 2+step_term_flag * 1;
    expected_pars = 8;
    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%expected_pars) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);
    phyper = (hypermet *) PyArray_DATA(param);

    if (nd_x == 0){
       *pret = 0;
        phyper = (hypermet *) PyArray_DATA(param);
        for (i=0;i<(npars/expected_pars);i++){
            /* g_term = st_term = lt_term = step_term = 0; */
            x1 = phyper[i].area;
            x2 = phyper[i].position;
            x3 = phyper[i].fwhm*tosigma;
            /* some intermediate variables */
            z0 = *px - x2;
            z1 = x3 * 1.4142135623730950488;
            /*I should check for sigma = 0 */
            if (x3 != 0) {
                z2 = (0.5 * z0 * z0) / (x3 * x3);
            }else{
                /* I should raise an exception */
                printf("Linear Algebra Error: Division by zero\n");
printf("Area=%f,Position=%f,FWHM=%f\n",x1,x2,phyper[i].fwhm);
printf("ST_Area=%f,ST_Slope=%f\n",phyper[i].st_area_r,phyper[i].st_slope_r);
printf("LT_Area=%f,LT_Slope=%f\n",phyper[i].lt_area_r,phyper[i].lt_slope_r);
                Py_DECREF(param);
                Py_DECREF(x);
                Py_DECREF(ret);
                return NULL;
            }
            if (g_term_flag){
                if (z2 < 612) {
                    *pret += exp (-z2) * (x1/(x3*sqrt2PI));
                }
            }
            if (st_term_flag){
                x4 = phyper[i].st_area_r;
                x5 = phyper[i].st_slope_r;
                if ((x5 != 0) && (x4 != 0)){
                    dhelp = x4 * 0.5 * erfc((z0/z1) + 0.5 * z1/x5);
                    if (dhelp != 0.0){
                    if (fabs(z0/x5) <= 612){
                *pret += ((x1 * dhelp)/x5) * exp(0.5 * (x3/x5) * (x3/x5)+ (z0/x5));
                    }
                    }
                }
            }
            if (lt_term_flag){
                x6 = phyper[i].lt_area_r;
                x7 = phyper[i].lt_slope_r;
                if ((x7 != 0) && (x6 != 0)){
                    dhelp = x6 * 0.5 * erfc((z0/z1) + 0.5 * z1/x7);
                    if (fabs(z0/x7) <= 612){
                *pret += ((x1 * dhelp)/x7) * exp(0.5 * (x3/x7) * (x3/x7)+(z0/x7));
                    }
                }
            }
            if (step_term_flag){
                x8 = phyper[i].step_height_r;
                if ((x8 != 0) && (x3 != 0)){
                *pret +=  x8 * (x1/(x3*sqrt2PI)) * 0.5 * erfc(z0/z1);
                }
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        for (j=0;j<k;j++){
            *pret = 0;
            phyper = (hypermet *) PyArray_DATA(param);
         for (i=0;i<(npars/expected_pars);i++){
            /* g_term = st_term = lt_term = step_term = 0; */
            x1 = phyper[i].area;
            x2 = phyper[i].position;
            x3 = phyper[i].fwhm * tosigma;
            /* some intermediate variables */
            z0 = *px - x2;
            z1 = x3 * 1.4142135623730950488;
            /*I should check for sigma = 0 */
            if (x3 != 0) {
                z2 = (0.5 * z0 * z0) / (x3 * x3);
            }else{
                /* I should raise an exception */
                printf("Linear Algebra Error: Division by zero\n");
printf("Area=%f,Position=%f,FWHM=%f\n",x1,x2,phyper[i].fwhm);
printf("ST_Area=%f,ST_Slope=%f\n",phyper[i].st_area_r,phyper[i].st_slope_r);
printf("LT_Area=%f,LT_Slope=%f\n",phyper[i].lt_area_r,phyper[i].lt_slope_r);
                Py_DECREF(param);
                Py_DECREF(x);
                Py_DECREF(ret);
                return NULL;
            }
            if (g_term_flag){
                if (z2 < 612) {
                    *pret += exp (-z2) * (x1/(x3*sqrt2PI));
                }
            }
            if (st_term_flag){
                x4 = phyper[i].st_area_r;
                x5 = phyper[i].st_slope_r;
                if ((x5 != 0) && (x4 != 0)){
                    dhelp = x4 * 0.5 * erfc((z0/z1) + 0.5 * z1/x5);
                    if (dhelp != 0){
                    if (fabs(z0/x5) <= 612){
                *pret += ((x1 * dhelp)/x5) * exp(0.5 * (x3/x5) * (x3/x5)+ (z0/x5));
                    }
                    }
                }
            }
            if (lt_term_flag){
                x6 = phyper[i].lt_area_r;
                x7 = phyper[i].lt_slope_r;
                if ((x7 != 0) && (x6 != 0)){
                    dhelp = x6 * 0.5 * erfc((z0/z1) + 0.5 * z1/x7);
                    if (fabs(z0/x7) <= 612){
                *pret += ((x1 * dhelp)/x7) * exp(0.5 * (x3/x7) * (x3/x7)+ (z0/x7));
                    }
                }
            }
            if (step_term_flag){
                x8 = phyper[i].step_height_r;
                if ((x8 != 0) && (x3 != 0)){
                *pret +=  x8 * (x1/(x3*sqrt2PI)) * 0.5 * erfc(z0/z1);
                }
            }
         }
            pret++;
            px++;
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}



double fastexp(double x)
{
int expindex;
static double EXP[5000] = {0.0};
int i;

/*initialize */
    if (EXP[0] < 1){
        for (i=0;i<5000;i++){
            EXP[i] = exp(-0.01 * i);
        }
    }
/*calculate*/
    if (x < 0){
        x = -x;
        if (x < 50){
            expindex = (int) (x * 100);
            return EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = (int) (x * 10);
            return pow(EXP[expindex]*(1.0 - (x - 0.1 * expindex)),10) ;
        }else if (x < 1000){
            expindex = (int) x;
            return pow(EXP[expindex]*(1.0 - (x - expindex)),20) ;
        }else if (x < 10000){
            expindex = (int) (x * 0.1);
            return pow(EXP[expindex]*(1.0 - (x - 10.0 * expindex)),30) ;
        }else{
            return 0;
        }
    }else{
        if (x < 50){
            expindex = (int) (x * 100);
            return 1.0/EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = (int) (x * 10);
            return pow(EXP[expindex]*(1.0 - (x - 0.1 * expindex)),-10) ;
        }else{
            return exp(x);
        }
    }
}


static PyObject *
SpecfitFuns_fastahypermet(PyObject *self, PyObject *args)
{
    double erfc(double);
    double fastexp(double);
    PyObject *input1, *input2;
    int debug=0;
    int tails=15;
    int expected_pars;
    int g_term_flag, st_term_flag, lt_term_flag, step_term_flag;
    /*double g_term, st_term, lt_term, step_term;*/
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    npy_intp dim_param[2];
    npy_intp dim_x[2];
    int i, j, k;
    double  dhelp, log2, sqrt2PI,tosigma;
    double x1, x2, x3, x4, x5, x6, x7, x8;
    double z0, z1, z2;
    double  *px, *pret;
    typedef struct {
        double  area;
        double  position;
        double  fwhm;
        double  st_area_r;
        double  st_slope_r;
        double  lt_area_r;
        double  lt_slope_r;
        double  step_height_r;
    } hypermet;
    hypermet *phyper;

    /** statements **/
    if (!PyArg_ParseTuple(args, "OO|ii", &input1,&input2,&tails,&debug))
        return NULL;

    param = (PyArrayObject *)
             PyArray_ContiguousFromObject(input1, NPY_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_ContiguousFromObject(input2, NPY_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }

    nd_param = PyArray_NDIM(param);
    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }
    if (nd_param == 1) {
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = PyArray_DIMS(param)[0];
        dim_param [1] = PyArray_DIMS(param)[1];
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }

    /* The gaussian terms must always be there */
    if(tails <= 0){
        /* I give back a matrix filled with zeros */
        ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
        if (ret == NULL){
            Py_DECREF(param);
            Py_DECREF(x);
            return NULL;
        }else{
            PyArray_FILLWBYTE(ret, 0);
            Py_DECREF(param);
            Py_DECREF(x);
            return PyArray_Return(ret);
        }
    }else{
        g_term_flag    = tails & 1;
        st_term_flag   = (tails>>1) & 1;
        lt_term_flag   = (tails>>2) & 1;
        step_term_flag = (tails>>3) & 1;
    }
    if (debug){
        printf("flags g = %d st = %d lt = %d step = %d\n",\
               g_term_flag,st_term_flag,lt_term_flag,step_term_flag);
    }
    expected_pars = 3 + st_term_flag * 2+lt_term_flag * 2+step_term_flag * 1;
    expected_pars = 8;
    if (nd_param == 1) {
        npars = (int) dim_param[0];
    }else{
        npars = (int) (dim_param[0] * dim_param[1]);
    }
    if ((npars%expected_pars) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n", (int)dim_param[0], (int)dim_param[1]);
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);
    phyper = (hypermet *) PyArray_DATA(param);

    if (nd_x == 0){
       *pret = 0;
        phyper = (hypermet *) PyArray_DATA(param);
        for (i=0;i<(npars/expected_pars);i++){
            /* g_term = st_term = lt_term = step_term = 0; */
            x1 = phyper[i].area;
            x2 = phyper[i].position;
            x3 = phyper[i].fwhm*tosigma;
            /* some intermediate variables */
            z0 = *px - x2;
            z1 = x3 * 1.4142135623730950488;
            /*I should check for sigma = 0 */
            if (x3 != 0) {
                z2 = (0.5 * z0 * z0) / (x3 * x3);
            }else{
                /* I should raise an exception */
                printf("Linear Algebra Error: Division by zero\n");
printf("Area=%f,Position=%f,FWHM=%f\n",x1,x2,phyper[i].fwhm);
printf("ST_Area=%f,ST_Slope=%f\n",phyper[i].st_area_r,phyper[i].st_slope_r);
printf("LT_Area=%f,LT_Slope=%f\n",phyper[i].lt_area_r,phyper[i].lt_slope_r);
                Py_DECREF(param);
                Py_DECREF(x);
                Py_DECREF(ret);
                return NULL;
            }
            if (z2 < 100) {
                if (g_term_flag){
                   /* *pret += exp (-z2) * (x1/(x3*sqrt2PI));*/
                   *pret += fastexp (-z2) * (x1/(x3*sqrt2PI));
                }
            }
            if (st_term_flag){
                x4 = phyper[i].st_area_r;
                x5 = phyper[i].st_slope_r;
                if ((x5 != 0) && (x4 != 0)){
                    dhelp = x4 * 0.5 * erfc((z0/z1) + 0.5 * z1/x5);
                    if (dhelp > 0.0){
                    if (fabs(z0/x5) <= 612){
              /*  *pret += ((x1 * dhelp)/x5) * exp(0.5 * (x3/x5) * (x3/x5)) \
                                      * exp(z0/x5);   */
                *pret += ((x1 * dhelp)/x5) * fastexp(0.5 * (x3/x5) * (x3/x5)+(z0/x5));
                    }
                    }
                }
            }
            if (lt_term_flag){
                x6 = phyper[i].lt_area_r;
                x7 = phyper[i].lt_slope_r;
                if ((x7 != 0) && (x6 != 0)){
                    dhelp = x6 * 0.5 * erfc((z0/z1) + 0.5 * z1/x7);
                    if (dhelp > 0.0){
                    if (fabs(z0/x7) <= 612){
                *pret += ((x1 * dhelp)/x7) * fastexp(0.5 * (x3/x7) * (x3/x7)+(z0/x7));
                    }
                    }
                }
            }
            if (step_term_flag){
                x8 = phyper[i].step_height_r;
                if ((x8 != 0) && (x3 != 0)){
                *pret +=  x8 * (x1/(x3*sqrt2PI)) * 0.5 * erfc(z0/z1);
                }
            }
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = (int) (dim_x [j] * k);
        }
        phyper = (hypermet *) PyArray_DATA(param);
        for (i=0;i<(npars/expected_pars);i++){
          if (i == 0){
                *pret = 0;
          }else{
            px = (double *) PyArray_DATA(x);
            pret = (double *) PyArray_DATA(ret);
          }
          x1 = phyper[i].area;
          x2 = phyper[i].position;
          x3 = phyper[i].fwhm * tosigma;
          x4 = phyper[i].st_area_r;
          x5 = phyper[i].st_slope_r;
          x6 = phyper[i].lt_area_r;
          x7 = phyper[i].lt_slope_r;
          x8 = phyper[i].step_height_r;
          z1 = x3 * 1.4142135623730950488;
          for (j=0;j<k;j++){
            /* some intermediate variables */
            z0 = *px - x2;
            /*I should check for sigma = 0 */
            if (x3 != 0) {
                z2 = (0.5 * z0 * z0) / (x3 * x3);
            }else{
                /* I should raise an exception */
                printf("Linear Algebra Error: Division by zero\n");
printf("Area=%f,Position=%f,FWHM=%f\n",x1,x2,phyper[i].fwhm);
printf("ST_Area=%f,ST_Slope=%f\n",phyper[i].st_area_r,phyper[i].st_slope_r);
printf("LT_Area=%f,LT_Slope=%f\n",phyper[i].lt_area_r,phyper[i].lt_slope_r);
                Py_DECREF(param);
                Py_DECREF(x);
                Py_DECREF(ret);
                return NULL;
            }
            if (z2 < 100){
            if (g_term_flag){
                   /* *pret += exp (-z2) * (x1/(x3*sqrt2PI));*/
                    *pret += fastexp (-z2) * (x1/(x3*sqrt2PI));
            }
            }
            /*include the short tail in the test is not a good idea */
            if (st_term_flag){
                if ((x5 != 0) && (x4 != 0)){
                    dhelp = (z0/z1) + 0.5 * z1/x5;
                    if (dhelp < 10){
                        dhelp = x4 * 0.5 * erfc(dhelp);
                        if (dhelp > 0){
                        if (fabs(z0/x5) <= 612){
                  *pret += ((x1 * dhelp)/x5) * fastexp(0.5 * (x3/x5) * (x3/x5) + (z0/x5));
                        }
                        }
                    }
                }
            }
            if (lt_term_flag){
                if ((x7 != 0) && (x6 != 0)){
                    dhelp = (z0/z1) + 0.5 * z1/x7;
                    if (dhelp < 10){
                        dhelp = x6 * 0.5 * erfc(dhelp);
                    if (dhelp > 0){
                        if (fabs(z0/x7) <= 612){
                *pret += ((x1 * dhelp)/x7) * fastexp(0.5 * (x3/x7) * (x3/x7)+(z0/x7));
                        }
                    }
                    }
                }
            }
            if (step_term_flag){
                if ((x8 != 0) && (x3 != 0)){
                *pret +=  x8 * (x1/(x3*sqrt2PI)) * 0.5 * erfc(z0/z1);
                }
            }

            pret++;
            px++;
         }
        }
    }

    Py_DECREF(param);
    Py_DECREF(x);
    return PyArray_Return(ret);
}



static PyObject *
SpecfitFuns_seek(PyObject *self, PyObject *args)
{
    long SpecfitFuns_seek2(long , long, long,
                  double, double, double,
                  long,
                  double, double, long,
                  double *, long, long, double *, long *, double *, double *);

    /* required input parameters */
    PyObject *input;        /* The array containing the y values */
    long    BeginChannel;   /* The first channel to start search */
    long    EndChannel;     /* The last channel of the search */
    double  FWHM;           /* The estimated FWHM in channels */

    /* optional input parameters */
    double  Sensitivity = 3.5;
    double  debug_info = 0;
    double  relevance_info = 0;

    /* some other variables required by the fortran function */
    long    FixRegion = 1;  /* Set to 1 if the program cannot adjust
                               the fitting region */
    double  LowDistance = 5.0;
    double  HighDistance = 3.0;
    long    AddInEmpty  = 0;
    long    npeaks;
    long    Ecal = 0;
    double  E[2];

    /* local variables */
    PyArrayObject    *yspec, *result;
    long        i;
    long        nchannels;
    long        NMAX_PEAKS = 150;
    double      peaks[150];
    double      relevances[150];
    long        seek_result;
    double      *pvalues;
    long        nd;
    npy_intp    dimensions[2];

    /* statements */
    if (!PyArg_ParseTuple(args, "Olld|ddd", &input, &BeginChannel,
                                    &EndChannel, &FWHM, &Sensitivity,
                                    &debug_info, &relevance_info ))
        return NULL;
    yspec = (PyArrayObject *)
             PyArray_CopyFromObject(input, NPY_DOUBLE,0,0);
    if (yspec == NULL)
        return NULL;
    if (Sensitivity < 0.1) {
        Sensitivity = 3.25;
    }

    nd = PyArray_NDIM(yspec);
    if (nd == 0) {
        printf("I need at least a vector!\n");
        Py_DECREF(yspec);
        return NULL;
    }

    nchannels = (long) PyArray_DIMS(yspec)[0];

    if (nd > 1) {
        if (nchannels == 1){
            nchannels = (long) PyArray_DIMS(yspec)[0];
        }
    }

    pvalues = (double *) PyArray_DATA(yspec);

    seek_result=SpecfitFuns_seek2(BeginChannel, EndChannel, nchannels,
                    FWHM, Sensitivity, debug_info,
                    FixRegion,
                    LowDistance, HighDistance, NMAX_PEAKS,
                    pvalues, AddInEmpty, Ecal,
                    E, &npeaks, peaks, relevances);



    Py_DECREF(yspec);

    if(seek_result != 0) {
        return NULL;
    }
    if (relevance_info) {
        dimensions [0] = npeaks;
        dimensions [1] = 2;
        result = (PyArrayObject *) PyArray_SimpleNew(2,dimensions,NPY_DOUBLE);
        pvalues = (double *) PyArray_DATA(result);
        for (i=0;i<npeaks;i++){
            pvalues[2*i] =  peaks[i];
            pvalues[2*i + 1] =  relevances[i];
        }
    }else{
        dimensions [0] = npeaks;
        result = (PyArrayObject *) PyArray_SimpleNew(1,dimensions,NPY_DOUBLE);
        pvalues = (double *) PyArray_DATA(result);
        for (i=0;i<npeaks;i++){
            pvalues[i] = peaks[i];
        }
    }
    return PyArray_Return(result);
}


long SpecfitFuns_seek2(long BeginChannel, long EndChannel,
      long nchannels, double FWHM, double Sensitivity,double debug_info,
      long FixRegion,
      double LowDistance, double HighDistance,long max_npeaks,
      double *yspec, long AddInEmpty, long Ecal,double *E,
      long *n_peaks, double *peaks, double *relevances)
{
    /* local variables */
    double  sigma, sigma2, sigma4;
    long    max_gfactor = 100;
    double  gfactor[100];
    long    nr_factor;
    double  sum_factors;
    double  lowthreshold;
    double  yspec2[2];
    double  nom;
    double  den2;
    long    begincalc, endcalc;
    long    channel1;
    long    lld;
    long    cch;
    long    cfac, cfac2;
    long    ihelp1, ihelp2;
    long    i, j;
    double  peakstarted = 0;

    /* statements */

    /* Make sure the peaks matrix is filled with zeros */
    for (i=0;i<max_npeaks;i++){
        peaks[i]      = 0.0;
        relevances[i] = 0.0;
    }

    /* prepare the calculation of the Gaussian scaling factors */

    sigma = FWHM / 2.35482;
    sigma2 = sigma * sigma;
    sigma4 = sigma2 * sigma2;
    lowthreshold = 0.01 / sigma2;
    sum_factors = 0.0;

    /* calculate the factors until lower threshold reached */
    j = MIN(max_gfactor, ((EndChannel - BeginChannel -2)/2)-1);
    for (cfac=1;cfac<j+1;cfac++) {
        cfac2 = cfac * cfac;
        gfactor[cfac-1] = (sigma2 - cfac2) * exp (-cfac2/(sigma2*2.0)) / sigma4;
        sum_factors += gfactor[cfac-1];
        /*printf("gfactor[%ld] = % g\n",cfac,gfactor[cfac-1]);*/
        if ((gfactor[cfac-1] < lowthreshold)
        && (gfactor[cfac-1] > (-lowthreshold))){
            break;
        }
    }
    /*printf("sum_factors = %g\n",sum_factors);*/
    nr_factor = cfac;

    /* What comes now is specific to MCA spectra ... */
    /*lld = 7;*/
    lld = 0;
    while (yspec [lld] == 0) {
        lld++;
    }
    lld = lld + (int) (0.5 * FWHM);

    channel1 = BeginChannel - nr_factor - 1;
    channel1 = MAX (channel1, lld);
    begincalc = channel1+nr_factor+1;
    endcalc = MIN (EndChannel+nr_factor+1, nchannels-nr_factor-1);
    *n_peaks = 0;
    cch = begincalc;
    if(debug_info){
        printf("nrfactor  = %ld\n", nr_factor);
        printf("begincalc = %ld\n", begincalc);
        printf("endcalc   = %ld\n", endcalc);
    }
    /* calculates smoothed value and variance at begincalc */
    cch = MAX(BeginChannel,0);
    nom = yspec[cch] / sigma2;
    den2 = yspec[cch] / sigma4;
    for (cfac = 1; cfac < nr_factor; cfac++){
        ihelp1 = cch-cfac;
        if (ihelp1 < 0){
            ihelp1 = 0;
        }
        ihelp2 = cch+cfac;
        if (ihelp2 >= nchannels){
            ihelp2 = nchannels-1;
        }
        nom += gfactor[cfac-1] * (yspec[ihelp2] + yspec [ihelp1]);
        den2 += gfactor[cfac-1] * gfactor[cfac-1] *
                 (yspec[ihelp2] + yspec [ihelp1]);
    }

    /* now normalize the smoothed value to the standard deviation */
    if (den2 <= 0.0) {
        yspec2[1] = 0.0;
    }else{
        yspec2[1] = nom / sqrt(den2);
    }
    yspec[0] = yspec[1];

    while (cch <= MIN(EndChannel,nchannels-2)){
        /* calculate gaussian smoothed values */
        yspec2[0] = yspec2[1];
        cch++;
        nom = yspec[cch]/sigma2;
        den2 = yspec[cch] / sigma4;
        for (cfac = 1; cfac < nr_factor; cfac++){
            ihelp1 = cch-cfac;
            if (ihelp1 < 0){
                ihelp1 = 0;
            }
            ihelp2 = cch+cfac;
            if (ihelp2 >= nchannels){
                ihelp2 = nchannels-1;
            }
            nom += gfactor[cfac-1] * (yspec[ihelp2] + yspec [ihelp1]);
            den2 += gfactor[cfac-1] * gfactor[cfac-1] *
                     (yspec[ihelp2] + yspec [ihelp1]);
        }
        /* now normalize the smoothed value to the standard deviation */
        if (den2 <= 0) {
            yspec2[1] = 0;
        }else{
            yspec2[1] = nom / sqrt(den2);
        }
/*        if (cch == endcalc)
            yspec2[1] = 0.0;

 */
        /* look if the current point falls in a peak */
        if (yspec2[1] > Sensitivity) {
            if(peakstarted == 0){
                if (yspec2[1] > yspec2[0]){
                    /* this second test is to prevent a peak from outside
                    the region from being detected at the beginning of the search */
                   peakstarted=1;
                }
            }
            /* there is a peak */
            if (debug_info){
                printf("At cch = %ld y[cch] = %g\n",cch,yspec[cch]);
                printf("yspec2[0] = %g\n",yspec2[0]);
                printf("yspec2[1] = %g\n",yspec2[1]);
                printf("Sensitivity = %g\n",Sensitivity);
            }
            if(peakstarted == 1){
                /* look for the top of the peak */
                if (yspec2[1] < yspec2 [0]) {
                    /* we are close to the top of the peak */
                    if (debug_info){
                        printf("we are close to the top of the peak\n");
                    }
                    if (*n_peaks < max_npeaks) {
                        peaks [*n_peaks] = cch-1;
                        relevances [*n_peaks] = yspec2[0];
                        (*n_peaks)++;
                        peakstarted=2;
                    }else{
                        printf("Found too many peaks\n");
                        return (-2);
                    }
                }
            }
            /* Doublet case */
            if(peakstarted == 2){
                if ((cch-peaks[(*n_peaks)-1]) > 0.6 * FWHM) {
                    if (yspec2[1] > yspec2 [0]){
                        if(debug_info){
                            printf("We may have a doublet\n");
                        }
                        peakstarted=1;
                    }
                }
            }
        }else{
            if (peakstarted==1){
            /* We were on a peak but we did not find the top */
                if(debug_info){
                    printf("We were on a peak but we did not find the top\n");
                }
            }
            peakstarted=0;
        }
    }
    if(debug_info){
      for (i=0;i< *n_peaks;i++){
        printf("Peak %ld found at ",i+1);
        printf("index %g with y = %g\n",peaks[i],yspec[(long ) peaks[i]]);
      }
    }

    return (0);
}

double myerfc(double x)
{
    double z;
    double t;
    double r;

    z=fabs(x);
    t=1.0/(1.0+0.5*z);
    r=t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.3740916 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223+t*0.17087277)))))))));
    if (x<0)
       r=2.0-r;
    return (r);
}

double myerf(double x)
{
    double z;
    double t;
    double r;

    z=fabs(x);
    t=1.0/(1.0+0.5*z);
    r=t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.3740916 +
      t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
      t * (1.48851587 + t * (-0.82215223+t*0.17087277)))))))));
    if (x<0)
       r=2.0-r;
    return (1.0-r);
}

double fasterfc(double x)
{
    double fasterf(double);
    return 1.0 - fasterf(x);

}
double fasterf(double z)
{
    /* error <= 3.0E-07 */
    double x;
    x=fabs(z);
    if (z>0){
        return 1.0 - pow(1.+ 0.0705230784 * x +
                         0.0422820123 * pow(x,2) +
                         0.0092705272 * pow(x,3) +
                         0.0001520143 * pow(x,4) +
                         0.0002765672 * pow(x,5) +
                         0.0000430638 * pow(x,6),-16);
    }else{
        return -1.0 + pow(1.+ 0.0705230784 * x +
                         0.0422820123 * pow(x,2) +
                         0.0092705272 * pow(x,3) +
                         0.0001520143 * pow(x,4) +
                         0.0002765672 * pow(x,5) +
                         0.0000430638 * pow(x,6),-16);
    }
}

static PyObject *
SpecfitFuns_interpol(PyObject *self, PyObject *args)
{
    /* required input parameters */
    PyObject *xinput;        /* The tuple containing the xdata arrays */
    PyObject *yinput;        /* The array containing the ydata values */
    PyObject *xinter0;       /* The array containing the x values */

    /* local variables */
    PyArrayObject    *ydata, *result, **xdata, *xinter;
    npy_intp i, j, k, l, jl, ju, offset, badpoint;
    double  value, *nvalue, *x1, *x2, *factors;
    double  dhelp, yresult;
    double  dummy = -1.0;
    npy_intp    nd_y, nd_x, index1, *points, *indices, max_points;
    /*int         dimensions[1];*/
    npy_intp npoints;
    npy_intp dimensions[2];
    npy_intp dim_xinter[2];
    double *helppointer;

    /* statements */
    if (!PyArg_ParseTuple(args, "OOO|d", &xinput, &yinput,&xinter0,&dummy)){
        printf("Parsing error\n");
        return NULL;
    }
    ydata = (PyArrayObject *)
             PyArray_CopyFromObject(yinput, NPY_DOUBLE,0,0);
    if (ydata == NULL){
        printf("Copy from Object error!\n");
        return NULL;
    }
    nd_y = PyArray_NDIM(ydata);
    if (nd_y == 0) {
        printf("I need at least a vector!\n");
        Py_DECREF(ydata);
        return NULL;
    }
/*
    for (i=0;i<nd_y;i++){
        printf("Dimension %d = %d\n",i,PyArray_DIMS(ydata)[i]);
    }
*/
    /* xdata parsing */
/*    (PyArrayObject *) xdata = (PyArrayObject *) malloc(nd_y * sizeof(PyArrayObject));*/
    xdata = (PyArrayObject **) malloc(nd_y * sizeof(PyArrayObject *));

    if (xdata == NULL){
        printf("Error in memory allocation\n");
        return NULL;
    }
    if (PySequence_Size(xinput) != nd_y){
        printf("xdata sequence of wrong length\n");
        return NULL;
    }
    for (i=0;i<nd_y;i++){
       /* printf("i = %d\n",i);*/
        /*xdata[i] = (PyArrayObject *)
                    PyArray_CopyFromObject(yinput,NPY_DOUBLE,0,0);
        */
        xdata[i] = (PyArrayObject *)
                    PyArray_CopyFromObject((PyObject *)
                    (PySequence_Fast_GET_ITEM(xinput,i)), NPY_DOUBLE,0,0);
        if (xdata[i] == NULL){
            printf("x Copy from Object error!\n");
            for (j=0;j<i;j++){
                Py_DECREF(xdata[j]);
            }
            free(xdata);
            Py_DECREF(ydata);
            return NULL;
        }
    }

    /* check x dimensions are appropriate */
    j=0;
    for (i=0;i<nd_y;i++){
        nd_x = PyArray_NDIM(xdata[i]);
        if (nd_x != 1) {
            printf("I need a vector!\n");
            j++;
            break;
        }
        if (PyArray_DIMS(xdata[i])[0] != PyArray_DIMS(ydata)[i]){
            printf("xdata[%d] does not have appropriate dimension\n", (int) i);
            j++;
            break;
        }
    }
    if (j) {
        for (i=0;i<nd_y;i++){
            Py_DECREF(xdata[i]);
        }
        free(xdata);
        Py_DECREF(ydata);
        return NULL;
    }

    xinter = (PyArrayObject *) PyArray_ContiguousFromObject(xinter0, NPY_DOUBLE,0,0);

    if (PyArray_NDIM(xinter) == 1){
        dim_xinter[0] = PyArray_DIMS(xinter)[0];
        dim_xinter[1] = 0;
        if (dim_xinter[0] != nd_y){
            printf("Wrong size\n");
            for (j=0;j<nd_y;j++){
                Py_DECREF(xdata[j]);
            }
            free(xdata);
            Py_DECREF(xinter);
            Py_DECREF(ydata);
            return NULL;
        }
    }else{
        dim_xinter[0] = PyArray_DIMS(xinter)[0];
        dim_xinter[1] = PyArray_DIMS(xinter)[1];
        if (dim_xinter[1] != nd_y){
            printf("Wrong size\n");
            for (j=0;j<nd_y;j++){
                Py_DECREF(xdata[j]);
            }
            free(xdata);
            Py_DECREF(xinter);
            Py_DECREF(ydata);
            return NULL;
        }
    }

    npoints = PyArray_DIMS(xinter)[0];
    helppointer = (double *) PyArray_DATA(xinter);
/*    printf("npoints = %d\n",npoints);
    printf("ndimensions y  = %d\n",nd_y);
*/
    /* Parse the points to interpolate */
    /* find the points to interpolate */
    max_points = 1;
    for (j=0; j< nd_y; j++){
        max_points = max_points * 2;
    }
    points  = malloc(max_points * nd_y * sizeof(npy_intp));
    indices = malloc(nd_y * sizeof(npy_intp));
    for (i=0;i<nd_y;i++){
        indices[i] = -1;
    }
    factors = malloc(nd_y * sizeof(double));
    dimensions [0] = npoints;

    result = (PyArrayObject *) PyArray_SimpleNew(1,dimensions,NPY_DOUBLE);

    for (i=0;i<npoints;i++){
        badpoint = 0;
        for (j=0; j< nd_y; j++){
            index1 = -1;
            if (badpoint == 0){
                value = *helppointer++;
                k=PyArray_DIMS(xdata[j])[0] - 1;
                nvalue = (double *) (PyArray_BYTES(xdata[j]) + k * (PyArray_STRIDES(xdata[j])[0]));
                /* test against other version
                valueold = PyFloat_AsDouble(
                    PySequence_Fast_GET_ITEM(PySequence_Fast_GET_ITEM(xinter0,i),j));
                if ( fabs(valueold-value) > 0.00001){
                    printf("i = %d, j= %d, oldvalue = %.5f, newvalue = %.5f\n",i,j,valueold, value);
                }
                */
                if (value > *nvalue){
                    badpoint = 1;
                }else{
                    nvalue = (double *) (PyArray_DATA(xdata[j]));
                    if (value < *nvalue){
                         badpoint = 1;
                    }
                }
                if (badpoint == 0){
                    if (1){
                        k = PyArray_DIMS(xdata[j])[0];
                        jl = -1;
                        ju = k-1;
                        if (badpoint == 0){
                            while((ju-jl) > 1){
                                k = (ju+jl)/2;
                                nvalue = (double *) (PyArray_BYTES(xdata[j]) + k * (PyArray_STRIDES(xdata[j])[0]));
                                if (value >= *nvalue){
                                    jl=k;
                                }else{
                                    ju=k;
                                }
                            }
                            index1=jl;
                        }
                    }
                    if (index1 < 0){
                        badpoint = 1;
                    }else{
                        x1 = (double *) (PyArray_BYTES(xdata[j])+ index1 * (PyArray_STRIDES(xdata[j])[0]));
                        x2 = (double *) (PyArray_BYTES(xdata[j])+(index1+1) * (PyArray_STRIDES(xdata[j])[0]));
                        factors[j] = (value - *x1) / (*x2 - *x1);
                        indices[j] = index1;
                    }
                }
            }else{
                helppointer++;
            }
        }
        if (badpoint == 1){
            yresult = dummy;
        }else{
          for (k=0;k<(max_points * nd_y);k++){
                j = k % nd_y;
                if (nd_y > 1){
                    l = k /(2 * (nd_y - j) );
                }else{
                    l = k;
                }
                if ( (l % 2 ) == 0){
                    points[k] = indices[j];
                }else{
                    points[k] = indices[j] + 1;
                }
             /*   printf("l = %d ,points[%d] = %d\n",l,k,points[k]);*/
          }
        /* the points to interpolate */
          yresult = 0.0;
          for (k=0;k<max_points;k++){
            dhelp =1.0;
            offset = 0;
            for (j=0;j<nd_y;j++){
                if (nd_y > 1){
                    l = ((nd_y * k) + j) /(2 * (nd_y - j) );
                }else{
                    l = ((nd_y * k) + j);
                }
                offset += points[(nd_y * k) + j] * (PyArray_STRIDES(ydata)[j]);
                /*printf("factors[%d] = %g\n",j,factors[j]);*/
                if ((l % 2) == 0){
                    dhelp = (1.0 - factors[j]) * dhelp;
                }else{
                    dhelp =  factors[j] * dhelp;
                }
            }
            yresult += *((double *) (PyArray_BYTES(ydata) + offset)) * dhelp;
          }
        }
       *((double *) (PyArray_BYTES(result) + i*PyArray_STRIDES(result)[0])) =  yresult;
    }
    free(points);
    free(indices);
    free(factors);
    for (i=0;i<nd_y;i++){
        Py_DECREF(xdata[i]);
    }
    free(xdata);
    Py_DECREF(ydata);
    Py_DECREF(xinter);

    return PyArray_Return(result);
}


static PyObject *
SpecfitFuns_voxelize(PyObject *self, PyObject *args)
{
    /* required input parameters */
    PyObject *grid_input;       /* The float array containing the float grid */
    PyObject *hits_input;       /* The int   array containing the number of hits */
    PyObject *xinput;     /* The tuple containing the double xdata arrays */
    PyObject *yinput;     /* The array containing the double ydata values */
    PyObject *limits_input;     /* The tuple containing the double xdata min and max values */
    int         use_datathreshold = 0;
    double   data_threshold = 0.0;

    /* local variables */
    PyArrayObject    *grid, *hits, *ydata, **xdata, *limits;

    npy_intp index;
    npy_intp *delta_index;
    npy_intp i, j, goodpoint, grid_position;

    double  value, limit0, limit1;
    npy_intp    nd_grid;
    npy_intp npoints;
    double  *data_pointer, *double_pointer;
    float   *grid_pointerf;
    double  *grid_pointerd;
    int     *hits_pointer;
    int        double_flag;

    /* statements */
    if (!PyArg_ParseTuple(args, "OOOOO|id", &grid_input, &hits_input, &xinput, &yinput, &limits_input,&use_datathreshold, &data_threshold)){
        printf("Parsing error\n");
        return NULL;
    }

    grid = (PyArrayObject *)
             PyArray_ContiguousFromObject(grid_input, NPY_NOTYPE,0,0);
    switch (PyArray_DESCR(grid)->type_num){
        case NPY_DOUBLE:
            double_flag = 1;
            break;
        default:
            double_flag = 0;
    }
    Py_DECREF(grid);
    if (double_flag){
        grid = (PyArrayObject *)
             PyArray_ContiguousFromObject(grid_input, NPY_DOUBLE,0,0);
    } else {
        grid = (PyArrayObject *)
             PyArray_ContiguousFromObject(grid_input, NPY_FLOAT,0,0);
    }
    nd_grid = PyArray_NDIM(grid);
    if (nd_grid == 0) {
        printf("Grid should be at least a vector!\n");
        Py_DECREF(grid);
        return NULL;
    }

    hits = (PyArrayObject *)
             PyArray_ContiguousFromObject(hits_input, NPY_INT,0,0);
    if (hits == NULL) {
        Py_DECREF(grid);
        return NULL;
    }


    if (PySequence_Size(xinput) != nd_grid){
        printf("xdata sequence of wrong length\n");
        Py_DECREF(grid);
        Py_DECREF(hits);
        return NULL;
    }

    if (PySequence_Size(limits_input) != (2*nd_grid)){
        printf("limits sequence of wrong length\n");
        Py_DECREF(grid);
        Py_DECREF(hits);
        return NULL;
    }

    ydata = (PyArrayObject *)
             PyArray_ContiguousFromObject(yinput, NPY_DOUBLE,0,0);
    if (ydata == NULL){
        printf("Contiguous from object error!\n");
        Py_DECREF(grid);
        Py_DECREF(hits);
        return NULL;
    }

    limits = (PyArrayObject *)
             PyArray_ContiguousFromObject(limits_input, NPY_DOUBLE,0,0);
    if (limits == NULL){
        printf("Limits. Contiguous from object error!\n");
        Py_DECREF(grid);
        Py_DECREF(ydata);
        Py_DECREF(hits);
        return NULL;
    }

    /* xdata parsing */
    xdata = (PyArrayObject **) malloc(nd_grid * sizeof(PyArrayObject *));
    if (xdata == NULL){
        printf("Error in memory allocation\n");
        Py_DECREF(grid);
        Py_DECREF(ydata);
        Py_DECREF(limits);
        Py_DECREF(hits);
        return NULL;
    }

    for (i=0;i<nd_grid;i++){
        xdata[i] = (PyArrayObject *)
                    PyArray_ContiguousFromObject((PyObject *)
                    (PySequence_Fast_GET_ITEM(xinput,i)), NPY_DOUBLE,0,0);
        if (xdata[i] == NULL){
            printf("x Copy from Object error!\n");
            for (j=0;j<i;j++){
                Py_DECREF(xdata[j]);
            }
            free(xdata);
            Py_DECREF(grid);
            Py_DECREF(ydata);
            Py_DECREF(limits);
            Py_DECREF(hits);
            return NULL;
        }
    }



    delta_index = (npy_intp *) malloc(nd_grid * sizeof(npy_intp));
    if (delta_index == NULL){
        printf("Error in memory allocation\n");
        Py_DECREF(grid);
        Py_DECREF(ydata);
        Py_DECREF(limits);
        free(xdata);
        Py_DECREF(hits);
        return NULL;
    }

    delta_index[nd_grid-1] = 1;
    for (i=0; i < nd_grid; i++){
        if (i==0){
            delta_index[nd_grid-1] = 1;
        }else{
            delta_index[nd_grid-1-i] = delta_index[nd_grid-i] * PyArray_DIMS(grid)[nd_grid-i];
        }
    }

    /* get the number of points in each of the arrays */
    npoints = 0;
    for (i=0; i<PyArray_NDIM(ydata);i++){
        if (i==0)
            npoints = PyArray_DIMS(ydata)[0];
        else
            npoints *= PyArray_DIMS(ydata)[i];
    }

    /* do the work */
    data_pointer = (double *) PyArray_DATA(ydata);
    grid_pointerf = (float *) PyArray_DATA(grid);
    grid_pointerd = (double *) PyArray_DATA(grid);
    hits_pointer = (int *) PyArray_DATA(hits);

    for (i=0;i<npoints;i++){
        if (use_datathreshold){
            if ((double) (*(data_pointer+i)) <= data_threshold)
                continue;
        }
        goodpoint = 1;
        grid_position = 0;
        for (j=0; j< nd_grid; j++){
            double_pointer = (double *) PyArray_DATA(xdata[j]);
            value = *(double_pointer+i);
            double_pointer = (double *) PyArray_DATA(limits);
            limit0 = *(double_pointer+j);
            limit1 = *(double_pointer+j+nd_grid);
            index = (int)(PyArray_DIMS(grid)[j]*(value - limit0)/(limit1-limit0));
            if ((index < 0) || (index >= PyArray_DIMS(grid)[j]))
            {
                /* this point is not going to contribute */
                goodpoint = 0;
                break;
            }
            grid_position += index * delta_index[j];
        }
        if (goodpoint){
            if (double_flag)
                *(grid_pointerd+grid_position) += (double) (*(data_pointer+i));
            else
                *(grid_pointerf+grid_position) += (float) (*(data_pointer+i));
            *(hits_pointer+grid_position) += 1;
        }
    }
    Py_DECREF(grid);
    Py_DECREF(hits);
    Py_DECREF(ydata);
    Py_DECREF(limits);
    for (i=0;i<nd_grid;i++){
        Py_DECREF(xdata[i]);
    }
    free(xdata);
    free(delta_index);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
SpecfitFuns_pileup(PyObject *self, PyObject *args)
{
    PyObject *input1;
    int    input2=0;
    double zero=0.0;
    double gain=1.0;
    int debug=0;
    PyArrayObject   *x;
    PyArrayObject   *ret;
    int nd_x;
    npy_intp dim_x[2];
    int i, j, k;
    double  *px, *pret, *pall;

    /** statements **/
    if (!PyArg_ParseTuple(args, "O|iddi", &input1, &input2, &zero, &gain, &debug))
        return NULL;

    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, NPY_DOUBLE,0,0);
    if (x == NULL)
       return NULL;

    nd_x = PyArray_NDIM(x);
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = PyArray_DIMS(x)[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;
        }else{
            dim_x [0] = PyArray_DIMS(x)[0];
            dim_x [1] = PyArray_DIMS(x)[1];
        }
    }
    if(debug !=0) {
        printf("x %d raws and %d cols\n", (int)dim_x[0], (int)dim_x[1]);
    }

    /* Create the output array */
    ret = (PyArrayObject *) PyArray_SimpleNew(nd_x, dim_x, NPY_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;
    }
    PyArray_FILLWBYTE(ret, 0);

    /* the pointer to the starting position of par data */
    px = (double *) PyArray_DATA(x);
    pret = (double *) PyArray_DATA(ret);

    if(1){
        *pret = 0;
        k = (int )(zero/gain);
        for (i=input2;i<dim_x[0];i++){
            pall=(double *) PyArray_DATA(x);
            if ((i+k) >= 0)
            {
                pret = (double *) PyArray_DATA(ret)+(i+k);
                for (j=0;j<dim_x[0]-i-k;j++){
                    *pret += *px * (*pall);
                    pall++;
                    pret++;
                }
            }
            px++;
        }
    }

    Py_DECREF(x);
    return PyArray_Return(ret);
}


static PyObject *
SpecfitFuns_SavitskyGolay(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject *ret;
    int n, npoints;
    double dpoints = 5.;
    double coeff[MAX_SAVITSKY_GOLAY_WIDTH];
    int i, j, m;
    double  dhelp, den;
    double  *data;
    double  *output;

    if (!PyArg_ParseTuple(args, "O|d", &input, &dpoints))
        return NULL;

    ret = (PyArrayObject *)
             PyArray_FROMANY(input, NPY_DOUBLE, 1, 1, NPY_ARRAY_ENSURECOPY);

    if (ret == NULL){
        printf("Cannot create 1D array from input\n");
        return NULL;
    }
    npoints = (int )  dpoints;
    if (!(npoints % 2)) npoints +=1;

    n = (int) PyArray_DIMS(ret)[0];

    if((npoints < MIN_SAVITSKY_GOLAY_WIDTH) ||  (n < npoints))
    {
        /* do not smooth data */
        return PyArray_Return(ret);
    }

    /* calculate the coefficients */
    m     = (int) (npoints/2);
    den = (double) ((2*m-1) * (2*m+1) * (2*m + 3));
    for (i=0; i<= m; i++){
        coeff[m+i] = (double) (3 * (3*m*m + 3*m - 1 - 5*i*i ));
        coeff[m-i] = coeff[m+i];
    }

    /* do the job */
    output = (double *) PyArray_DATA(ret);

    /* simple smoothing at the beginning */
    for (j=0; j<=(int)(npoints/3); j++)
    {
        smooth1d(output, m);
    }

    /* simple smoothing at the end */
    for (j=0; j<=(int)(npoints/3); j++)
    {
        smooth1d((output+n-m-1), m);
    }

    /*one does not need the whole spectrum buffer, but code is clearer */
    data = (double *) malloc(n * sizeof(double));
    memcpy(data, output, n * sizeof(double));

    /* the actual SG smoothing in the middle */
    for (i=m; i<(n-m); i++){
        dhelp = 0;
        for (j=-m;j<=m;j++) {
            dhelp += coeff[m+j] * (*(data+i+j));
        }
        if(dhelp > 0.0){
            *(output+i) = dhelp / den;
        }
    }
    free(data);
    return PyArray_Return(ret);

}

/* List of functions defined in the module */

static PyMethodDef SpecfitFuns_methods[] = {
    {"snip1d",      SpecfitFuns_snip1d,     METH_VARARGS},
    {"snip2d",      SpecfitFuns_snip2d,     METH_VARARGS},
    {"snip3d",      SpecfitFuns_snip3d,     METH_VARARGS},
    {"subacold",    SpecfitFuns_subacold,   METH_VARARGS},
    {"subac",       SpecfitFuns_subac,      METH_VARARGS},
    {"subacfast",   SpecfitFuns_subacfast,  METH_VARARGS},
    {"gauss",       SpecfitFuns_gauss,      METH_VARARGS},
    {"agauss",      SpecfitFuns_agauss,     METH_VARARGS},
    {"fastagauss",  SpecfitFuns_fastagauss, METH_VARARGS},
    {"alorentz",    SpecfitFuns_alorentz,   METH_VARARGS},
    {"lorentz",     SpecfitFuns_lorentz,    METH_VARARGS},
    {"apvoigt",     SpecfitFuns_apvoigt,    METH_VARARGS},
    {"pvoigt",      SpecfitFuns_pvoigt,     METH_VARARGS},
    {"downstep",    SpecfitFuns_downstep,   METH_VARARGS},
    {"upstep",      SpecfitFuns_upstep,     METH_VARARGS},
    {"slit",        SpecfitFuns_slit,       METH_VARARGS},
    {"ahypermet",   SpecfitFuns_ahypermet,  METH_VARARGS},
    {"fastahypermet",   SpecfitFuns_fastahypermet,  METH_VARARGS},
    {"erfc",        SpecfitFuns_erfc,       METH_VARARGS},
    {"erf",         SpecfitFuns_erf,        METH_VARARGS},
    {"seek",        SpecfitFuns_seek,       METH_VARARGS},
    {"interpol",    SpecfitFuns_interpol,   METH_VARARGS},
    {"voxelize",    SpecfitFuns_voxelize,   METH_VARARGS},
    {"pileup",      SpecfitFuns_pileup,   METH_VARARGS},
    {"SavitskyGolay",   SpecfitFuns_SavitskyGolay,   METH_VARARGS},
    {"splitgauss",  SpecfitFuns_splitgauss,   METH_VARARGS},
    {"splitlorentz",SpecfitFuns_splitlorentz, METH_VARARGS},
    {"splitpvoigt", SpecfitFuns_splitpvoigt, METH_VARARGS},
    {NULL,        NULL}        /* sentinel */
};

/* ------------------------------------------------------- */


/* Module initialization */

#if PY_MAJOR_VERSION >= 3

static int SpecfitFuns_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int SpecfitFuns_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "SpecfitFuns",
        NULL,
        sizeof(struct module_state),
        SpecfitFuns_methods,
        NULL,
        SpecfitFuns_traverse,
        SpecfitFuns_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_SpecfitFuns(void)

#else
#define INITERROR return

void
initSpecfitFuns(void)
#endif
{
    struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("SpecfitFuns", SpecfitFuns_methods);
#endif

    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("SpecfitFuns.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
