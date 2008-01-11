#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
#include <Python.h>
#include <./numpy/arrayobject.h>
#include <math.h>
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


static PyObject *ErrorObject;

typedef struct {
    PyObject_HEAD
    PyObject    *x_attr;    /* Attributes dictionary */
} SpecfitFunsObject;

staticforward PyTypeObject SpecfitFuns_Type;

/*
 * Function prototypes
 */
static void                SpecfitFuns_dealloc  (SpecfitFunsObject *self);

#define SpecfitFunsObject_Check(v)    ((v)->ob_type == &SpecfitFuns_Type)

/* SpecfitFunso methods */

static void
SpecfitFuns_dealloc(self)
    SpecfitFunsObject *self;
{
    Py_XDECREF(self->x_attr);
    PyObject_DEL(self);
}


static int
SpecfitFuns_setattr(SpecfitFunsObject *self, char *name,
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
                    "delete non-existing SpecfitFuns attribute");
        return rv;
    }
    else
        return PyDict_SetItemString(self->x_attr, name, v);
}

/* --------------------------------------------------------------------- */

/* Function SUBAC returning smoothed array */

static PyObject *
SpecfitFuns_subacold(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *array, *ret;
    int n, dimensions[1];
    double niter0 = 5000.;
    int i, j, niter = 5000;
    double  t_old, t_mean, c = 1.0000;
    double  *data;
    
    if (!PyArg_ParseTuple(args, "O|dd", &input, &c, &niter0))
        return NULL;
    array = (PyArrayObject *)
             PyArray_CopyFromObject(input, PyArray_DOUBLE,1,1);
    if (array == NULL)
        return NULL;
    niter = (int ) niter0;
    n = array->dimensions[0];
    dimensions[0] = array->dimensions[0];
    ret = (PyArrayObject *)
        PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(array);
        return NULL;
    }

    /* Do the job */
    data = (double *) array->data;
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
    ret = (PyArrayObject *) PyArray_Copy(array);
    Py_DECREF(array);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);  

}

static PyObject *
SpecfitFuns_subac(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *array, *ret, *anchors;
    int n, dimensions[1];
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
    array = (PyArrayObject *)
             PyArray_CopyFromObject(input, PyArray_DOUBLE,1,1);
    if (array == NULL)
        return NULL;
    deltai= (int ) deltai0;
    if (deltai <=0) deltai = 1;
    niter = (int ) niter0;
    n = array->dimensions[0];
    dimensions[0] = array->dimensions[0];
    ret = (PyArrayObject *)
        PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(array);
        return NULL;
    }
    memcpy(ret->data, array->data, array->dimensions[0] * sizeof(double));

    if (n < (2*deltai+1)){
        /*ret = (PyArrayObject *) PyArray_Copy(array);*/
        Py_DECREF(array);
        return PyArray_Return(ret);
    }
    /* do the job */
    data   = (double *) array->data;
    retdata   = (double *) ret->data;

    if (anchors0 != NULL)
    {
        if (PySequence_Check(anchors0)){
            anchors = (PyArrayObject *)
                 PyArray_ContiguousFromObject(anchors0, PyArray_INT, 1, 1);
            if (anchors == NULL)
            {
                Py_DECREF(array);
                Py_DECREF(ret);
                return NULL;
            }
            anchordata = (int *) anchors->data;
            nanchors   = PySequence_Size(anchors0);
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
                memcpy(array->data, ret->data, array->dimensions[0] * sizeof(double));
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
            memcpy(array->data, ret->data, array->dimensions[0] * sizeof(double));
        }
    }
    Py_DECREF(array);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);  

}

static PyObject *
SpecfitFuns_subacfast(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject   *array, *ret, *anchors;
    int n, dimensions[1];
    double niter0 = 5000.;
    double deltai0= 1;
    PyObject *anchors0 = NULL;
    int i, j, k, l, deltai = 1,niter = 5000;
    double  t_mean, c = 1.000;
    double  *data, *retdata;
    int     *anchordata;
    int nanchors, notdoit;

    if (!PyArg_ParseTuple(args, "O|dddO", &input, &c, &niter0,&deltai0, &anchors0))
        return NULL;
    array = (PyArrayObject *)
             PyArray_CopyFromObject(input, PyArray_DOUBLE,1,1);
    if (array == NULL)
        return NULL;
    deltai= (int ) deltai0;
    if (deltai <=0) deltai = 1;
    niter = (int ) niter0;
    n = array->dimensions[0];
    dimensions[0] = array->dimensions[0];
    ret = (PyArrayObject *)
        PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(array);
        return NULL;
    }
    memcpy(ret->data, array->data, array->dimensions[0] * sizeof(double));

    if (n < (2*deltai+1)){
        /*ret = (PyArrayObject *) PyArray_Copy(array);*/
        Py_DECREF(array);
        return PyArray_Return(ret);
    }
    /* do the job */
    data   = (double *) array->data;
    retdata   = (double *) ret->data;
    if (PySequence_Check(anchors0)){
        anchors = (PyArrayObject *)
             PyArray_ContiguousFromObject(anchors0, PyArray_INT, 1, 1);
        if (anchors == NULL)
    {
            Py_DECREF(array);
            Py_DECREF(ret);
            return NULL;
        }
    anchordata = (int *) anchors->data;
        nanchors   = PySequence_Size(anchors0);
        memcpy(array->data, ret->data, array->dimensions[0] * sizeof(double));
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
        memcpy(array->data, ret->data, array->dimensions[0] * sizeof(double));
        for (i=0;i<niter;i++){
            for (j=deltai;j<n-deltai;j++) {
                t_mean = 0.5 * (*(retdata+j-deltai) + *(retdata+j+deltai));
            if (*(retdata+j) > (t_mean * c))
                    *(retdata+j) = t_mean;
            }
        }
    }
    Py_DECREF(array);
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
    int dim_param[2];
    int dim_x[2];
    int i, j, k;
    double  dhelp, log2;
    double  *px, *pret;
    char *tpe;
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
             PyArray_ContiguousFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    log2 = 0.69314718055994529;
    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) param->data;
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
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            pgauss = (gaussian *) param->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) param->data;
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
            k = dim_x [j] * k;
        }
        pgauss = (gaussian *) param->data;
        for (i=0;i<(npars/3);i++){
            sigma  = pgauss[i].fwhm*tosigma;                
            dhelp0 = pgauss[i].area/(sigma*sqrt2PI);
            px = (double *) x->data;
            pret = (double *) ret->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        pgauss = (gaussian *) param->data;
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
            k = dim_x [j] * k;
        }
        pgauss = (gaussian *) param->data;
        for (i=0;i<(npars/3);i++){
            sigma  = pgauss[i].fwhm*tosigma;                
            dhelp0 = pgauss[i].area/(sigma*sqrt2PI);
            px = (double *) x->data;
            pret = (double *) ret->data;
            for (j=0;j<k;j++){
                if (i==0)
                    *pret = 0.0;
                dhelp = (*px - pgauss[i].centroid)/sigma;
                if (dhelp <= 15){
                    dhelp = 0.5 * dhelp * dhelp;
                    if (dhelp < 50){
                        expindex = dhelp * 100;
                        *pret += dhelp0 * EXP[expindex]*(1.0 - (dhelp - 0.01 * expindex)) ;
                    }else if (dhelp < 100) {
                        expindex = dhelp * 10;
              *pret += dhelp0 * pow(EXP[expindex]*(1.0 - (dhelp - 0.1 * expindex)),10) ;                     
                    }else if (dhelp < 1000){
                        expindex = dhelp;
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
SpecfitFuns_apvoigt(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        ppvoigt = (pvoigtian *) param->data;
        for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * \
                (ppvoigt[i].area / (0.5 * M_PI * ppvoigt[i].fwhm * dhelp));
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            ppvoigt = (pvoigtian *) param->data;
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
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
        ppvoigt = (pvoigtian *) param->data;
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
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            ppvoigt = (pvoigtian *) param->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        ppvoigt = (pvoigtian *) param->data;
        for (i=0;i<(npars/4);i++){
            dhelp = (*px - ppvoigt[i].centroid) / (0.5 * ppvoigt[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += ppvoigt[i].eta * (ppvoigt[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            ppvoigt = (pvoigtian *) param->data;
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
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
        ppvoigt = (pvoigtian *) param->data;
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
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            ppvoigt = (pvoigtian *) param->data;
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
SpecfitFuns_lorentz(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        plorentz = (lorentzian *) param->data;
        for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += (plorentz[i].height / dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            plorentz = (lorentzian *) param->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       *pret = 0;
        plorentz = (lorentzian *) param->data;
        for (i=0;i<(npars/3);i++){
            dhelp = (*px - plorentz[i].centroid) / (0.5 * plorentz[i].fwhm);
            dhelp = 1.0 + (dhelp * dhelp);
            *pret += plorentz[i].area /(0.5 * M_PI * plorentz[i].fwhm * dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            plorentz = (lorentzian *) param->data;
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
SpecfitFuns_downstep(PyObject *self, PyObject *args)
{
    double erfc(double);
    PyObject *input1, *input2;
    int debug=0;
    PyArrayObject   *param, *x;
    PyArrayObject   *ret;
    int nd_param, nd_x, npars;
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));


    if (nd_x == 0){
       *pret = 0;
        perrorfc = (errorfc *) param->data;
        for (i=0;i<(npars/3);i++){        
            dhelp = perrorfc[i].fwhm * tosigma;
            dhelp = (*px - perrorfc[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorfc[i].height * 0.5 * erfc(dhelp);
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorfc = (errorfc *) param->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%3) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));

    if (nd_x == 0){
       *pret = 0;
        perrorf = (errorf *) param->data;
        for (i=0;i<(npars/3);i++){        
            dhelp = perrorf[i].fwhm * tosigma;
            dhelp = (*px - perrorf[i].centroid) / (sqrt(2)*dhelp);
            *pret += perrorf[i].height * 0.5 * (1.0 + erf(dhelp));
        }
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorf = (errorf *) param->data;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }

    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if (nd_param == 1) {
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%4) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    tosigma=1.0/(2.0*sqrt(2.0*log(2.0)));

    if (nd_x == 0){
       *pret = 0;
        perrorf = (errorf *) param->data;
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
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            perrorf = (errorf *) param->data;
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
    int dim_x[2];
    int j, k;
    double  dhelp;
    double  *px, *pret;


    /** statements **/
    if (!PyArg_ParseTuple(args, "O|i", &input1,&debug))
        return NULL;
        
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (x == NULL){
        return NULL;
    }    

    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if(debug !=0) {
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       dhelp = *px;
       *pret = erfc(dhelp);
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
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
    int dim_x[2];
    int j, k;
    double  dhelp;
    double  *px, *pret;


    /** statements **/
    if (!PyArg_ParseTuple(args, "O|i", &input1,&debug))
        return NULL;
        
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (x == NULL){
        return NULL;
    }    

    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    if(debug !=0) {
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if (nd_x == 0){
       dhelp = *px;
       *pret = erf(dhelp);
    }else{
        k = 1;
        for (j=0;j<nd_x;j++){
            k = dim_x [j] * k;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_CopyFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }
    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    /* The gaussian terms must always be there */
    if(tails <= 0){
        /* I give back a matrix filled with zeros */
        ret = (PyArrayObject *)
        PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
        if (ret == NULL){
            Py_DECREF(param);
            Py_DECREF(x);
            return NULL;    
        }else{
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
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%expected_pars) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;
    phyper = (hypermet *) param->data;
if(0){
if(debug !=0){
    for (i=0;i<(npars/expected_pars);i++){       
printf("Area%d=%f,Pos%d=%f,FWHM%d=%f\n",i,phyper[i].area,i,phyper[i].position,i,phyper[i].fwhm);
    }
}
}
    if (nd_x == 0){
       *pret = 0;
        phyper = (hypermet *) param->data;
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
            k = dim_x [j] * k;
        }
        for (j=0;j<k;j++){
            *pret = 0;
            phyper = (hypermet *) param->data;
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
static double EXP[5000];
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
            expindex = x * 100;
            return EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = x * 10;
            return pow(EXP[expindex]*(1.0 - (x - 0.1 * expindex)),10) ;                     
        }else if (x < 1000){
            expindex = x;
            return pow(EXP[expindex]*(1.0 - (x - expindex)),20) ;
        }else if (x < 10000){
            expindex = x * 0.1;
            return pow(EXP[expindex]*(1.0 - (x - 10.0 * expindex)),30) ;
        }else{
            return 0;
        }
    }else{
        if (x < 50){
            expindex = x * 100;
            return 1.0/EXP[expindex]*(1.0 - (x - 0.01 * expindex)) ;
        }else if (x < 100) {
            expindex = x * 10;
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
    int dim_param[2];
    int dim_x[2];
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
             PyArray_ContiguousFromObject(input1, PyArray_DOUBLE,0,0);
    if (param == NULL)
        return NULL;
    x = (PyArrayObject *)
             PyArray_ContiguousFromObject(input2, PyArray_DOUBLE,0,0);
    if (x == NULL){
        Py_DECREF(param);
        return NULL;
    }    

    nd_param = param->nd;
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_param = %d nd_x = %d\n",nd_param,nd_x);
    }
    if (nd_param == 1) {
        dim_param [0] = param->dimensions[0];
        dim_param [1] = 0;
    }else{
        dim_param [0] = param->dimensions[0];
        dim_param [1] = param->dimensions[1];
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }

    /* The gaussian terms must always be there */
    if(tails <= 0){
        /* I give back a matrix filled with zeros */
        ret = (PyArrayObject *)
        PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
        if (ret == NULL){
            Py_DECREF(param);
            Py_DECREF(x);
            return NULL;    
        }else{
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
        npars = dim_param[0];
    }else{
        npars = dim_param[0] * dim_param[1];
    }
    if ((npars%expected_pars) != 0) {
        printf("Incorrect number of parameters\n");
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;
    }

    if(debug !=0) {
        printf("parameters %d raws and %d cols\n",dim_param[0],dim_param[1]); 
        printf("nparameters = %d\n",npars);
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(param);
        Py_DECREF(x);
        return NULL;    
    }

    log2 = 0.69314718055994529;
    sqrt2PI= sqrt(2.0*M_PI);
    tosigma=1.0/(2.0*sqrt(2.0*log2));

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;
    phyper = (hypermet *) param->data;
if(0){
if(debug !=0){
    for (i=0;i<(npars/expected_pars);i++){       
printf("Area%d=%f,Pos%d=%f,FWHM%d=%f\n",i,phyper[i].area,i,phyper[i].position,i,phyper[i].fwhm);
    }
}
}
    if (nd_x == 0){
       *pret = 0;
        phyper = (hypermet *) param->data;
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
            k = dim_x [j] * k;
        }
        phyper = (hypermet *) param->data;
        for (i=0;i<(npars/expected_pars);i++){
          if (i == 0){
                *pret = 0;
          }else{            
            px = (double *) x->data;
            pret = (double *) ret->data;
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
                    if (dhelp <3){
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
                    if (dhelp < 3){
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
    long        i, nchannels; 
    long        NMAX_PEAKS = 100;
    double      peaks[100];
    double      relevances[100];
    long        seek_result;    
    double      *pvalues;
    long        nd;
    int         dimensions[2];
    
    /* statements */        
    if (!PyArg_ParseTuple(args, "Olld|ddd", &input, &BeginChannel,
                                    &EndChannel, &FWHM, &Sensitivity,
                                    &debug_info, &relevance_info ))
        return NULL;
    yspec = (PyArrayObject *)
             PyArray_CopyFromObject(input, PyArray_DOUBLE,0,0);
    if (yspec == NULL)
        return NULL;
    if (Sensitivity < 0.1) {
        Sensitivity = 3.25;
    }

    nd = yspec->nd;
    if (nd == 0) {
        printf("I need at least a vector!\n");
        Py_DECREF(yspec);
        return NULL;
    }

    nchannels = yspec->dimensions[0];

    if (nd > 1) {
        if (nchannels == 1){
            nchannels = yspec->dimensions[0];
        }
    }

    pvalues = (double *) yspec->data;

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
        result = (PyArrayObject *) 
             PyArray_FromDims(2,dimensions,PyArray_DOUBLE);
        for (i=0;i<npeaks;i++){
            /*printf("Peak %ld found at %g rel %g\n",i+1,peaks[i],relevances[i]);*/
            *((double *) (result->data + i*result->strides[0])) =  peaks[i];
            *((double *) (result->data +(i*result->strides[0] + result->strides[1]))) =  relevances[i];
        }
        
    }else{
        dimensions [0] = npeaks;
        result = (PyArrayObject *) 
             PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
        for (i=0;i<npeaks;i++){
            /*printf("Peak %ld found at %g\n",i+1,peaks[i]);*/
            *((double *) (result->data +i*result->strides[0])) =  peaks[i];
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
    double  distance;
    long    i, j;
    double  peakstarted = 0;

    /* statements */
    distance=MAX(LowDistance, HighDistance);
    
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
    lld = lld + 0.5 * FWHM;
    
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
                printf("At cch = %ld y[cch] = %g ",cch,yspec[cch]);
                printf("yspec2[0] = %g ",yspec2[0]);
                printf("yspec2[1] = %g ",yspec2[1]);
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
    long    i, j, k, l, jl, ju, offset, badpoint; 
    double  value, *nvalue, *x1, *x2, *factors;
    double  dhelp, yresult;
    double  dummy = -1.0;
    long    nd_y, nd_x, index1, npoints, *points, *indices;
    /*int         dimensions[1];*/
    int     dimensions[1];
    int     dim_xinter[1];
    double *helppointer;
    
    /* statements */        
    if (!PyArg_ParseTuple(args, "OOO|d", &xinput, &yinput,&xinter0,&dummy)){
        printf("Parsing error\n");
        return NULL;
    }
    ydata = (PyArrayObject *)
             PyArray_CopyFromObject(yinput, PyArray_DOUBLE,0,0);
    if (ydata == NULL){
        printf("Copy from Object error!\n");    
        return NULL;
    }
    nd_y = ydata->nd;
    if (nd_y == 0) {
        printf("I need at least a vector!\n");
        Py_DECREF(ydata);
        return NULL;
    }
/*
    for (i=0;i<nd_y;i++){
        printf("Dimension %d = %d\n",i,ydata->dimensions[i]);    
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
                    PyArray_CopyFromObject(yinput,PyArray_DOUBLE,0,0);
        */
        xdata[i] = (PyArrayObject *)
                    PyArray_CopyFromObject((PyObject *)
                    (PySequence_Fast_GET_ITEM(xinput,i)), PyArray_DOUBLE,0,0);
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
        nd_x = xdata[i]->nd;
        if (nd_x != 1) {
            printf("I need a vector!\n");
            j++;
            break;
        }
        if (xdata[i]->dimensions[0] != ydata->dimensions[i]){
            printf("xdata[%ld] does not have appropriate dimension\n",i);
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

    xinter = (PyArrayObject *) PyArray_ContiguousFromObject(xinter0, PyArray_DOUBLE,0,0);
    
    if (xinter->nd == 1){
        dim_xinter[0] = xinter->dimensions[0];
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
        dim_xinter[0] = xinter->dimensions[0];
        dim_xinter[1] = xinter->dimensions[1];
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

    npoints = xinter->dimensions[0];
    helppointer = (double *) xinter->data;
/*    printf("npoints = %d\n",npoints);
    printf("ndimensions y  = %d\n",nd_y);
*/
    /* Parse the points to interpolate */    
    /* find the points to interpolate */
    points  = malloc(pow(2,nd_y) * nd_y * sizeof(int));
    indices = malloc(nd_y * sizeof(int));
    for (i=0;i<nd_y;i++){
        indices[i] = -1;
    }
    factors = malloc(nd_y * sizeof(double));
    dimensions [0] = npoints;
    
    result = (PyArrayObject *) 
             PyArray_FromDims(1,dimensions,PyArray_DOUBLE);

    for (i=0;i<npoints;i++){
        badpoint = 0;
        for (j=0; j< nd_y; j++){
            index1 = -1;
            if (badpoint == 0){
                value = *helppointer++;
                k=xdata[j]->dimensions[0] - 1;
                nvalue = (double *) (xdata[j]->data + k * xdata[j]->strides[0]);
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
                    nvalue = (double *) (xdata[j]->data);
                    if (value < *nvalue){
                         badpoint = 1;
                    }
                }
                if (badpoint == 0){
                    if (1){
                        k = xdata[j]->dimensions[0];
                        jl = -1;
                        ju = k-1;
                        if (badpoint == 0){
                            while((ju-jl) > 1){
                                k = (ju+jl)/2;
                                nvalue = (double *) (xdata[j]->data + k * xdata[j]->strides[0]);
                                if (value >= *nvalue){
                                    jl=k;
                                }else{
                                    ju=k;                    
                                }            
                            }
                            index1=jl;
                        }
                    }
                    /* test against the other version */
                    if (0){
                        k=0;
                        ju = -1;
                        while(k < (xdata[j]->dimensions[0] - 1)){
                            nvalue = (double *) (xdata[j]->data + k * xdata[j]->strides[0]);
                            /*printf("nvalue = %g\n",*nvalue);*/
                            if (value >= *nvalue){
                                ju = k;
                             }
                            k++;
                        }
                        if (ju != index1){
                            printf("i = %ld, j= %ld, value = %.5f indexvalue = %ld, newvalue = %ld\n",i,j,value,ju, index1);
                        }
                    }
                    if (index1 < 0){
                        badpoint = 1;
                    }else{
                        x1 = (double *) (xdata[j]->data + index1 * xdata[j]->strides[0]);
                        x2 = (double *) (xdata[j]->data + (index1+1) * xdata[j]->strides[0]);
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
          for (k=0;k<(pow(2,nd_y) * nd_y);k++){
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
          for (k=0;k<pow(2,nd_y);k++){
            dhelp =1.0;
            offset = 0;
            for (j=0;j<nd_y;j++){
                if (nd_y > 1){
                    l = ((nd_y * k) + j) /(2 * (nd_y - j) );
                }else{
                    l = ((nd_y * k) + j);
                }
                offset += points[(nd_y * k) + j] * (ydata -> strides[j]);
                /*printf("factors[%d] = %g\n",j,factors[j]);*/
                if ((l % 2) == 0){ 
                    dhelp = (1.0 - factors[j]) * dhelp;
                }else{
                    dhelp =  factors[j] * dhelp;
                }
            }
            yresult += *((double *) (ydata -> data + offset)) * dhelp;
          }
        }
       *((double *) (result->data +i*result->strides[0])) =  yresult;
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
    int dim_x[2];
    int i, j, k;
    double  *px, *pret, *pall;

    /** statements **/
    if (!PyArg_ParseTuple(args, "O|iddi", &input1, &input2, &zero, &gain, &debug))
        return NULL;

    x = (PyArrayObject *)
             PyArray_CopyFromObject(input1, PyArray_DOUBLE,0,0);
    if (x == NULL)
       return NULL;
        
    nd_x = x->nd;
    if(debug !=0) {
        printf("nd_x = %d\n",nd_x);
    }

    if (nd_x == 1) {
        dim_x [0] = x->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (nd_x == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = x->dimensions[0];
            dim_x [1] = x->dimensions[1];
        }
    }
    if(debug !=0) {
        printf("x %d raws and %d cols\n",dim_x[0],dim_x[1]); 
    }

    /* Create the output array */
    ret = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(x);
        return NULL;    
    }

    /* the pointer to the starting position of par data */
    px = (double *) x->data;
    pret = (double *) ret->data;

    if(1){
        *pret = 0;
        k = (int )(zero/gain);
        for (i=input2;i<dim_x[0];i++){
            pall=(double *) x->data;
            if ((i+k) >= 0) 
            {
                pret = (double *) ret->data+(i+k);
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
    PyArrayObject   *array, *ret;
    int n, npoints, dimensions[1];
    double dpoints = 5.;
    double coeff[MAX_SAVITSKY_GOLAY_WIDTH];     
    int i, j, m;
    double  dhelp, den;
    double  *data;
    double  *output;
    
    if (!PyArg_ParseTuple(args, "O|d", &input, &dpoints))
        return NULL;
    array = (PyArrayObject *)
             PyArray_ContiguousFromObject(input, PyArray_DOUBLE,1,1);
    if (array == NULL)
        return NULL;
    npoints = (int )  dpoints;
    if (!(npoints % 2)) npoints +=1;
    n = array->dimensions[0];
    dimensions[0] = array->dimensions[0];
    ret = (PyArrayObject *)
        PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
    if (ret == NULL){
        Py_DECREF(array);
        return NULL;
    }

    if((npoints < MIN_SAVITSKY_GOLAY_WIDTH) ||  (n < npoints)){
        /* do not smooth data */
        /* ret = (PyArrayObject *) PyArray_Copy(array); */
        memcpy(ret->data, array->data, array->dimensions[0] * sizeof(double));
        Py_DECREF(array);
        return PyArray_Return(ret);
    }
    /* calculate the coefficients */
    m     = (int) (npoints/2);
    den = (double) ((2*m-1) * (2*m+1) * (2*m + 3)); 
    for (i=0; i<= m; i++){
        coeff[m+i] = (double) (3 * (3*m*m + 3*m - 1 - 5*i*i ));
        coeff[m-i] = coeff[m+i];
    }
    /*
    for (i=0;i<npoints;i++)
        printf("m = %d,%d %f\n",m, i, coeff[i]);    
    printf("denominator %f\n",den);
    */

    /* do the job */
    data   = (double *) array->data;
    output = (double *) ret->data;
    
    for (i=0; i<m; i++){
        *(output+i) = *(data+i);
    }
    for (i=m; i<(n-m); i++){
        dhelp = 0;
        for (j=-m;j<=m;j++) {
            dhelp += coeff[m+j] * (*(data+i+j));
        }
        if(dhelp > 0.0){
            *(output+i) = dhelp / den;
        }else{
            *(output+i) = *(data+i);
        }
    }
    for (i=(n-m); i<n; i++){
        *(output+i) = *(data+i);
    }
    
    Py_DECREF(array);
    if (ret == NULL)
        return NULL;
    return PyArray_Return(ret);  

}


static PyObject *
SpecfitFuns_spline(PyObject *self, PyObject *args)
{
    /* required input parameters */              
    PyObject *xinput;        /* The tuple containing the xdata arrays */
    PyObject *yinput;        /* The array containing the ydata values */
    PyObject *xinter0;       /* The array containing the x values */

    /* local variables */    
    PyArrayObject    *xdata, *ydata, *result, *uarray;
    int dim_x[2], nd_x, nd_y;
    int n,i,k;
    double p, qn, sig, un, *u, *y2, *x, *y;
    
    /* statements */        
    if (!PyArg_ParseTuple(args, "OO|O", &xinput, &yinput,&xinter0)){
        printf("Parsing error\n");
        return NULL;
    }
    xdata = (PyArrayObject *)
             PyArray_CopyFromObject(xinput, PyArray_DOUBLE,0,0);
    if (xdata == NULL){
        printf("Copy from X Object error!\n");    
        return NULL;
    }
    nd_x = xdata->nd;
    if (nd_x != 1) {
        printf("I need a X vector!\n");
        Py_DECREF(xdata);
        return NULL;
    }
    ydata = (PyArrayObject *)
             PyArray_CopyFromObject(yinput, PyArray_DOUBLE,0,0);
    if (ydata == NULL){
        printf("Copy from Y Object error!\n");    
        return NULL;
    }
    nd_y = ydata->nd;
    if (nd_y != 1) {
        printf("I need a Y vector!\n");
        Py_DECREF(ydata);
        return NULL;
    }
    if (xdata->dimensions[0] != ydata->dimensions[0]){
        printf("X and Y do not have same dimension!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        return NULL;       
    }
    /* build the output array */
    dim_x [0] = xdata->dimensions[0];
    dim_x [1] = 0;
    result = (PyArrayObject *)
    PyArray_FromDims(nd_x, dim_x, PyArray_DOUBLE);
    if (result == NULL){
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        return NULL;    
    }
    
    /* build the temporary array */
    uarray = (PyArrayObject *)
             PyArray_Copy(result);
    if (uarray == NULL){
        printf("Copy from result Object error!\n");    
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(result);        
        return NULL;
    }

    /* the pointer to the starting position of par data */
    x   = (double *) ( xdata->data);
    y   = (double *) ( ydata->data);
    y2  = (double *) (result->data);
    u   = (double *) (uarray->data);

    y2[0] = u[0] = 0.0;
    n = xdata->dimensions[0];
      for (i=1;i<=(n-2);i++) {
        /*printf("i = [%d] x = %f, y = %f\n",i,x[i],y[i]);*/
        sig=(x[i] - x[i-1])/( x[i+1] - x[i-1]);
        p=sig * y2[i-1]+2.0;
        y2[i]=(sig-1.0)/p;
        u[i]=(y[i+1]-y[i])/( x[i+1] - x[i]) - (y[i]-y[i-1])/( x[i] - x[i-1]);
        u[i]=(6.0*u[i]/(x[i+1] - x[i-1])-sig*u[i-1])/p;
    }
    
    qn=un=0.0;
    y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);    
    for (k=n-2;k>=0;k--)
        y2[k]=y2[k]*y2[k+1]+u[k];
    
    Py_DECREF(xdata);
    Py_DECREF(ydata);
    Py_DECREF(uarray);
    
    return PyArray_Return(result);
}

static PyObject *
SpecfitFuns_splint(PyObject *self, PyObject *args)
{
    /* required input parameters */              
    PyObject *xinput ;        /* The tuple containing the xdata arrays */
    PyObject *yinput ;        /* The array containing the ydata values */
    PyObject *y2input;        /* The array containing the y2data values */
    PyObject *xinter0;        /* The array containing the x values */

    /* local variables */    
    PyArrayObject    *xdata, *ydata,  *y2data, *xinter, *result;
    int dim_x[2], nd_x, nd_y;
    int n;
    double *xa, *ya, *y2a, *x, *y;
    int klo, khi, k, i, j;
    double h, b, a;
    
    /* statements */        
    if (!PyArg_ParseTuple(args, "OOOO", &xinput, &yinput, &y2input,&xinter0)){
        printf("Parsing error\n");
        return NULL;
    }
    xdata = (PyArrayObject *)
             PyArray_CopyFromObject(xinput, PyArray_DOUBLE,0,0);
    if (xdata == NULL){
        printf("Copy from X Object error!\n");    
        return NULL;
    }
    nd_x = xdata->nd;
    if (nd_x != 1) {
        printf("I need a X vector!\n");
        Py_DECREF(xdata);
        return NULL;
    }
    ydata = (PyArrayObject *)
             PyArray_CopyFromObject(yinput, PyArray_DOUBLE,0,0);
    if (ydata == NULL){
        printf("Copy from Y Object error!\n");    
        return NULL;
    }
    nd_y = ydata->nd;
    if (nd_y != 1) {
        printf("I need a Y vector!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        return NULL;
    }
    y2data = (PyArrayObject *)
             PyArray_CopyFromObject(y2input, PyArray_DOUBLE,0,0);
    if (y2data == NULL){
        printf("Copy from Y2 Object error!\n"); 
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        return NULL;
    }
    if (y2data->nd != 1) {
        printf("I need a Y2 vector!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(y2data);
        return NULL;
    }
    if (xdata->dimensions[0] != ydata->dimensions[0]){
        printf("X and Y do not have same dimension!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(y2data);
        return NULL;       
    }
    if (xdata->dimensions[0] != y2data->dimensions[0]){
        printf("X and Y2 do not have same dimension!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(y2data);
        return NULL;       
    }


    xinter =  (PyArrayObject *) PyArray_ContiguousFromObject(xinter0, PyArray_DOUBLE,0,0);   
    if (xinter == NULL){
        printf("Copy from x error!\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(y2data);
        return NULL;   
    }
    /* build the output array */
    if (xinter->nd == 1) {
        dim_x [0] = xinter->dimensions[0];
        dim_x [1] = 0;
    }else{
        if (xinter->nd == 0) {
            dim_x [0] = 0;
            dim_x [1] = 0;        
        }else{
            dim_x [0] = xinter->dimensions[0];
            dim_x [1] = xinter->dimensions[1];
        }
    }

    /* Create the output array */
    result = (PyArrayObject *) PyArray_Copy(xinter);
    if (result == NULL){
        printf("Cannot build result array\n");
        Py_DECREF(xdata);
        Py_DECREF(ydata);
        Py_DECREF(y2data);
        Py_DECREF(xinter);
        return NULL;
    }
    xa  = (double *)    xdata->data;
    ya  = (double *)    ydata->data;
    y2a = (double *)   y2data->data;
    x   = (double *)   xinter->data;
    y   = (double *)   result->data;
    n = xdata->dimensions[0];
    /*printf("xdata ->dimensions[0] = %d\n",n);
    printf("xinter->nd = %d\n",xinter->nd);*/
    if (xinter->nd == 0){
        klo=0;
        khi=n-1;
        while (khi-klo > 1) {
            k=(khi+klo) >> 1;
            if (xa[k] > x[0]) khi=k;
            else klo=k;
        }
        h=xa[khi]-xa[klo];
        /*printf("xinter = %f\n, xlow= %f , xhigh = %f \n",x[0],xa[klo],xa[khi]);
        printf("interpolation between %d and % d, h = % f\n",klo,khi,h);*/
        if (h == 0.0) {
            printf("Bad table input, repeated values\n");
            Py_DECREF(xdata);
            Py_DECREF(ydata);
            Py_DECREF(y2data);
            Py_DECREF(xinter);
            Py_DECREF(result);           
        }
        a=(xa[khi]-x[0])/h;
        b=(x[0]-xa[klo])/h;
        *y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
    }else{
        j = 1;
        for (i=0;i<(xinter->nd);i++){
            j = j * xinter->dimensions[i];
        }
        for (i=0; i<j;i++){
            klo=0;
            khi=n-1;
            while (khi-klo > 1) {
                k=(khi+klo) >> 1;
                if (xa[k] > x[i]) khi=k;
                else klo=k;
            }
            h=xa[khi]-xa[klo];
            if (h == 0.0) {
                printf("Bad table input, repeated values\n");
                Py_DECREF(xdata);
                Py_DECREF(ydata);
                Py_DECREF(y2data);
                Py_DECREF(xinter);
                Py_DECREF(result);           
            }
            a=(xa[khi]-x[i])/h;
            b=(x[i]-xa[klo])/h;
            *y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
            y++;
        }    
    }
    
    Py_DECREF(xdata);
    Py_DECREF(ydata);
    Py_DECREF(y2data);
    Py_DECREF(xinter);
    
    return PyArray_Return(result);
}

/* List of functions defined in the module */

static PyMethodDef SpecfitFuns_methods[] = {
    {"subacold",    SpecfitFuns_subacold,    METH_VARARGS},
    {"subac",        SpecfitFuns_subac,        METH_VARARGS},
    {"subacfast",        SpecfitFuns_subacfast,        METH_VARARGS},
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
    {"pileup",      SpecfitFuns_pileup,   METH_VARARGS},
    {"SavitskyGolay",      SpecfitFuns_SavitskyGolay,   METH_VARARGS},
    {"spline",      SpecfitFuns_spline,   METH_VARARGS},
    {"_splint",     SpecfitFuns_splint,   METH_VARARGS},
    {NULL,        NULL}        /* sentinel */
};


static PyObject *
SpecfitFuns_getattr(SpecfitFunsObject *self,
    char *name)
{
    if (self->x_attr != NULL) {
        PyObject *v = PyDict_GetItemString(self->x_attr, name);
        if (v != NULL) {
            Py_INCREF(v);
            return v;
        }
    }
    return Py_FindMethod(SpecfitFuns_methods, (PyObject *)self, name);
}


statichere PyTypeObject SpecfitFuns_Type = {
    /* The ob_type field must be initialized in the module init function
     * to be portable to Windows without using C++. */
    PyObject_HEAD_INIT(NULL)
    0,            /*ob_size*/
    "SpecfitFuns",                /*tp_name*/
    sizeof(SpecfitFunsObject),    /*tp_basicsize*/
    0,            /*tp_itemsize*/
    /* methods */
    (destructor)SpecfitFuns_dealloc, /*tp_dealloc*/
    0,            /*tp_print*/
    (getattrfunc)SpecfitFuns_getattr, /*tp_getattr*/
    (setattrfunc)SpecfitFuns_setattr, /*tp_setattr*/
    0,            /*tp_compare*/
    0,            /*tp_repr*/
    0,            /*tp_as_number*/
    0,            /*tp_as_sequence*/
    0,            /*tp_as_mapping*/
    0,            /*tp_hash*/
};


/* Initialization function for the module (*must* be called initSpecfitFuns) */

DL_EXPORT(void)
initSpecfitFuns(void)
{
    PyObject *m, *d;

    /* Initialize the type of the new type object here; doing it here
     * is required for portability to Windows without requiring C++. */
    SpecfitFuns_Type.ob_type = &PyType_Type;

    /* Create the module and add the functions */
    m = Py_InitModule("SpecfitFuns", SpecfitFuns_methods);
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    ErrorObject = PyErr_NewException("SpecfitFuns.error", NULL, NULL);
    PyDict_SetItemString(d, "error", ErrorObject);
}
