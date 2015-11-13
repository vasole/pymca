#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>

#include <./numpy/arrayobject.h>

#include "libqhull.h"
#include "qset.h"        /* for FOREACHneighbor_() */
#include "poly.h"        /* for qh_vertexneighbors() */
#include "geom.h"       /* for qh_facetcenter() */


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif


/* Doc strings */
#if (REALfloat == 1)
PyDoc_STRVAR(Object3DQhull__doc__,
"Object3DQhullf is just an interface module to the Qhull library.\n"
"    For the time being only delaunay triangulation is implemented.\n"
"    See http://www.qhull.org for Qhull details.\n"
"\n"
"Object3DQHullf.delaunay(nodes, \"qhull  d Qbb QJ Qc Po\")\n"
"    Nodes is a sequence of points (an nrows x 2 or an nrows x 3 array)\n"
"    The second argument is optional.\n"
"    The output is an array of indices for the facets.\n");
PyDoc_STRVAR(Object3DQhull_delaunay__doc__,
"delaunay(nodes, \"qhull  d Qbb QJ Qc Po\")\n"
"    Nodes is a sequence of points (an nrows x 2 or an nrows x 3 array)\n"
"    The second argument is optional.\n"
"    http://www.qhull.org for Qhull details.\n"
"    The output is an array of indices for the facets.\n");
#else
PyDoc_STRVAR(Object3DQhull__doc__,
"Object3DQhull is just an interface module to the Qhull library.\n"
"    For the time being only delaunay triangulation is implemented.\n"
"    See http://www.qhull.org for Qhull details.\n"
"\n"
"Object3DQHull.delaunay(nodes, \"qhull  d Qbb QJ Qc\")\n"
"    Nodes is a sequence of points (an nrows x 2 or an nrows x 3 array)\n"
"    The second argument is optional.\n"
"    The output is an array of indices for the facets.\n");
PyDoc_STRVAR(Object3DQhull_delaunay__doc__,
"delaunay(nodes, \"qhull  d Qbb QJ Qc\")\n"
"    Nodes is a sequence of points (an nrows x 2 or an nrows x 3 array)\n"
"    The second argument is optional.\n"
"    http://www.qhull.org for Qhull details.\n"
"    The output is an array of indices for the facets.\n");
#endif


/* Function declarations */
static PyObject *object3DDelaunay(PyObject *dummy, PyObject *args);
static PyObject *object3DVoronoi(PyObject *dummy, PyObject *args);
static void qhullResultFailure(PyObject * self, int qhull_exitcode);
static PyObject *getQhullVersion(PyObject *dummy, PyObject *args);


static PyObject *object3DDelaunay(PyObject *self, PyObject *args)
{
    /* input parameters */
    PyObject    *input1, *input3=NULL;
    const char      *input2 = NULL;

    /* local variables */
    PyArrayObject    *pointArray, *inner_pointArray=NULL;
    PyArrayObject    *result, *inner_result=NULL ;

    coordT    *points;    /* Qhull */
    int        dimension;    /* Qhull */
    int        nPoints;    /* Qhull */
    int        inner_nPoints = 0;    /* Qhull */

    int        qhullResult;        /* Qhull exit code, 0 means no error */
    boolT ismalloc = False;        /* True if Qhull should free points in
                                   qh_freeqhull() or reallocation */
    //char cQhullDefaultFlags[] = "qhull d Qbb Qt"; /* Qhull flags (see doc)*/
#if (REALfloat == 1)
    char cQhullDefaultFlags[] = "qhull d Qbb QJ Qc Po"; /* Qhull flags (see doc) Po is to ignore precision errors*/
#else
    char cQhullDefaultFlags[] = "qhull d Qbb QJ Qc"; /* Qhull flags (see doc)*/
#endif
    char *cQhullFlags;

    int            nFacets = 0;
    npy_intp    outDimensions[3];
    facetT *facet;        /* needed by FORALLfacets */
    vertexT *vertex, **vertexp;
    int j, i;
#if (REALfloat == 1)
    float *p;
    float bestdist;
    float point[4];
#else
    double *p;
    double bestdist;
    double point[4];
#endif
    unsigned int *uintP;
    boolT isoutside;
    struct module_state *st = GETSTATE(self);



    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "O|zO", &input1, &input2, &input3 ))
    {
        PyErr_SetString(st->error, "Unable to parse arguments");
        return NULL;
    }

    /* The array containing the points */
#if (REALfloat == 1)
    pointArray = (PyArrayObject *)
        PyArray_ContiguousFromAny(input1, PyArray_FLOAT,2,2);


    if(input3) {
        inner_pointArray = (PyArrayObject *)
            PyArray_ContiguousFromAny(input3, PyArray_FLOAT,2,2);
      if(!inner_pointArray) {
        PyErr_SetString(st->error, "third argument if given must be  a nrows x X array");
        return NULL;
      }
    }
#else

    pointArray = (PyArrayObject *)
                    PyArray_ContiguousFromAny(input1, PyArray_DOUBLE,2,2);
    if(input3) {

      inner_pointArray = (PyArrayObject *)
        PyArray_ContiguousFromAny(input3, PyArray_DOUBLE,2,2);
      if(!inner_pointArray) {
        PyErr_SetString(st->error, "third argument if given must be  a nrows x X array");
        return NULL;
      }
    }
#endif


    if (pointArray == NULL)
    {
        PyErr_SetString(st->error, "First argument is not a nrows x X array");
        return NULL;
    }
    if (input2 == NULL)
      {
    cQhullFlags = &cQhullDefaultFlags[0];
      }
    else
      {
    cQhullFlags = (char *) input2;
      }
    /* printf("flags = %s\n", cQhullFlags); */

    /* dimension to pass to Qhull */
    dimension = (int) pointArray->dimensions[1];

    /* number of points for Qhull */
    nPoints = (int) pointArray->dimensions[0];


    /* the points themselves for Qhull */
    points = (coordT *) pointArray->data;

    qhullResult = qh_new_qhull(dimension, nPoints, points,
                   ismalloc, cQhullFlags, NULL, stderr);

    if (qhullResult)
    {
        /* Free the memory allocated by Qhull */
        qh_freeqhull(qh_ALL);
        Py_DECREF (pointArray);
        if(input3) {
            Py_DECREF (inner_pointArray);
        }
        qhullResultFailure(self, qhullResult);
        return NULL;
    }

    /* Get the number of facets */
    /* Probably there is a better way to do it */
    FORALLfacets {
        if (facet->upperdelaunay)
            continue;
        nFacets ++;
    }
    /* printf("Number of facets = %d\n", nFacets); */

    /* Allocate the memory for the output array */
    if (0)    // As triangles
    {
        /* It has the form: [nfacets, dimension, 3] */
        outDimensions[0] = nFacets;
        outDimensions[1] = 3;
        outDimensions[2] = dimension;
        result = (PyArrayObject *)
          PyArray_SimpleNew(3, outDimensions, PyArray_FLOAT);
        if (result == NULL)
        {
            qh_freeqhull(qh_ALL);
            Py_DECREF (pointArray);
            if(input3) {
                Py_DECREF (inner_pointArray);
            }
            PyErr_SetString(st->error, "Error allocating output memory");
            return NULL;
        }
#if (REALfloat == 1)
        p = (float *) result->data;
#else
        p = (double *) result->data;
#endif
        FORALLfacets {
            if (facet->upperdelaunay)
                continue;
            FOREACHvertex_(facet->vertices)    {
                for (j = 0; j < (qh hull_dim - 1); ++j) {
                    *p =  vertex->point[j];
                    ++p;
                }
            }
        }
    }
    else // As indices
    {

        outDimensions[0] = nFacets;
        outDimensions[1] =   dimension+1  ;
        result = (PyArrayObject *)
                        PyArray_SimpleNew(2, outDimensions, PyArray_UINT32);

        if( input3 )
        {
            inner_nPoints = (int) inner_pointArray->dimensions[0];

            outDimensions[0] = inner_nPoints;
            outDimensions[1] = dimension+1;
            inner_result = (PyArrayObject *)
            PyArray_SimpleNew(2, outDimensions, PyArray_UINT32);
            if (inner_result == NULL)
            {
                qh_freeqhull(qh_ALL);
                Py_DECREF (pointArray);
                Py_DECREF (inner_pointArray);
                PyErr_SetString(st->error, "Error allocating output memory for inner points facets");
                return NULL;
            }
        }

        if (result == NULL)
        {
            qh_freeqhull(qh_ALL);
            Py_DECREF (pointArray);
            {
                if(inner_pointArray)
                {
                    Py_DECREF (inner_pointArray) ;
                }
            }
            PyErr_SetString(st->error, "Error allocating output memory");
            return NULL;
        }

        uintP = (unsigned int *) result->data;
        FORALLfacets {
            if (facet->upperdelaunay)
                continue;
            FOREACHvertex_(facet->vertices)    {
                *uintP =  qh_pointid(vertex->point);
                ++uintP;
            }
        }
        if(input3)
        {
            uintP = (unsigned int *) inner_result->data;
#if (REALfloat == 1)
            p = (float *) inner_pointArray->data;
#else
            p = (double *) inner_pointArray->data;
#endif
            for (i=0; i< inner_nPoints; i++)
            {
                for(j=0; j<dimension; j++)
                {
                    point[j] = *( p++);
                }

                qh_setdelaunay(  dimension+1 , 1, point);

                facet = qh_findbestfacet(point, qh_ALL, &bestdist, &isoutside);

                if (facet && !facet->upperdelaunay && facet->simplicial)
                {
                    FOREACHvertex_(facet->vertices)    {
                        *uintP =  qh_pointid(vertex->point);
                        ++uintP;
                    }
                }
                else
                {
                    qh_freeqhull(qh_ALL);
                    Py_DECREF (pointArray);
                    if(pointArray)
                    {
                        Py_DECREF (inner_pointArray);
                    }
                    PyErr_SetString(st->error, "Error allocating output memory");
                    return NULL;
                }
            }
        }
    }


    /* Free the memory allocated by Qhull */
    qh_freeqhull(qh_ALL);

    Py_DECREF (pointArray);

    if( input3==NULL)
    {
        return PyArray_Return(result);
    }
    else
    {
        return  Py_BuildValue("NN", result, inner_result );
    }
}

static PyObject *object3DVoronoi(PyObject *self, PyObject *args)
{
    /* input parameters */
    PyObject    *input1, *input3=NULL;
    const char      *input2 = NULL;

    /* local variables */
    PyArrayObject    *pointArray, *inner_pointArray=NULL;
    PyArrayObject    *result;

    coordT    *points;    /* Qhull */
    int        dimension;    /* Qhull */
    int        nPoints;    /* Qhull */

    int        qhullResult;        /* Qhull exit code, 0 means no error */
    boolT ismalloc = False;        /* True if Qhull should free points in
                                   qh_freeqhull() or reallocation */
#if (REALfloat == 1)
    char cQhullDefaultFlags[] = "qhull v p"; /* Qhull flags (see doc) Po is to ignore precision errors*/
#else
    char cQhullDefaultFlags[] = "qhull v p"; /* Qhull flags (see doc)*/
#endif
    char *cQhullFlags;

    int            nFacets = 0;
    npy_intp    outDimensions[2];
    facetT *facet;        /* needed by FORALLfacets */
    pointT *center;

    int j, i;
#if (REALfloat == 1)
    float *p;
#else
    double *p;
#endif
    struct module_state *st = GETSTATE(self);

    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "O|zO", &input1, &input2, &input3 ))
    {
        PyErr_SetString(st->error, "Unable to parse arguments");
        return NULL;
    }

    /* The array containing the points */
#if (REALfloat == 1)
    pointArray = (PyArrayObject *)
        PyArray_ContiguousFromAny(input1, PyArray_FLOAT,2,2);


    if(input3) {
        inner_pointArray = (PyArrayObject *)
            PyArray_ContiguousFromAny(input3, PyArray_FLOAT,2,2);
      if(!inner_pointArray) {
        PyErr_SetString(st->error, "third argument if given must be  a nrows x X array");
        return NULL;
      }
    }
#else

    pointArray = (PyArrayObject *)
                    PyArray_ContiguousFromAny(input1, PyArray_DOUBLE,2,2);
    if(input3) {

      inner_pointArray = (PyArrayObject *)
        PyArray_ContiguousFromAny(input3, PyArray_DOUBLE,2,2);
      if(!inner_pointArray) {
        PyErr_SetString(st->error, "third argument if given must be  a nrows x X array");
        return NULL;
      }
    }
#endif


    if (pointArray == NULL)
    {
        PyErr_SetString(st->error, "First argument is not a nrows x X array");
        return NULL;
    }
    if (input2 == NULL)
      {
    cQhullFlags = &cQhullDefaultFlags[0];
      }
    else
      {
    cQhullFlags = (char *) input2;
      }
    /* printf("flags = %s\n", cQhullFlags); */

    /* dimension to pass to Qhull */
    dimension = (int) pointArray->dimensions[1];

    /* number of points for Qhull */
    nPoints = (int) pointArray->dimensions[0];


    /* the points themselves for Qhull */
    points = (coordT *) pointArray->data;

    qhullResult = qh_new_qhull(dimension, nPoints, points,
                   ismalloc, cQhullFlags, NULL, stderr);

    if (qhullResult)
      {
        /* Free the memory allocated by Qhull */
        qh_freeqhull(qh_ALL);
        Py_DECREF (pointArray);
        if(input3) {
            Py_DECREF (inner_pointArray);
        }
        qhullResultFailure(self, qhullResult);
        return NULL;
      }
    /* Get the number of facets */
    /* Probably there is a better way to do it */
    i = 0;
    FORALLfacets {
      if (facet->upperdelaunay)
    continue;
      i += 1;
      printf("Facet number %d\n", i);
      nFacets ++;
    }
    printf("Number of facets = %d\n", nFacets);

    /* Allocate the memory for the output array */
    /* It has the form: [nfacets, dimension, 3] */
    outDimensions[0] = nFacets;
    outDimensions[1] = dimension;
    //outDimensions[2] = dimension;
    printf("output dimensions = %ld, %ld\n", outDimensions[0], outDimensions[1]);
    result = (PyArrayObject *)
      PyArray_SimpleNew(2, outDimensions, PyArray_DOUBLE);
    if (result == NULL)
      {
        qh_freeqhull(qh_ALL);
        Py_DECREF (pointArray);
        if(input3) {
            Py_DECREF (inner_pointArray);
        }
        PyErr_SetString(st->error, "Error allocating output memory");
        return NULL;
      }
#if (REALfloat == 1)
    printf("FLOAT\n");
    p = (float *) result->data;
#else
    p = (double *) result->data;
    printf("DOUBLE\n");
#endif

    printf("qh hull_dim = %d\n", qh hull_dim);
    i = 0;
    if (1)
    {
        FORALLfacets {
          if (facet->upperdelaunay)
              continue;
          if (facet->visitid > 0)
          {
              i += 1;
          center = qh_facetcenter(facet->vertices);
          for (j = 0; j < (qh hull_dim - 1); ++j) {
              printf("vertex[%d] =  %f\n", j, center[j]);
              //p = ((double *) result->data) + (facet->visitid-1) * dimension + j;
              *p =  center[j];
              p++;
          }
          }
        }
        printf("Number of Voronoi vertices = %d\n", i);
    }
    printf("PASSED LOOP\n");


    /* Free the memory allocated by Qhull */
    qh_freeqhull(qh_ALL);

    Py_DECREF (pointArray);

    return PyArray_Return(result);
}


static void
qhullResultFailure(PyObject * self, int qhull_exitcode)
{
    struct module_state *st = GETSTATE(self);

    switch (qhull_exitcode) {
    case qh_ERRinput:
        PyErr_BadInternalCall ();
        break;
    case qh_ERRsingular:
        PyErr_SetString(PyExc_ArithmeticError,
                "qhull singular input data");
        break;
    case qh_ERRprec:
        PyErr_SetString(PyExc_ArithmeticError,
                "qhull precision error");
        break;
    case qh_ERRmem:
        PyErr_NoMemory();
        break;
    case qh_ERRqhull:
        PyErr_SetString(st->error,
                "qhull internal error");
        break;
    }
}

static PyObject *getQhullVersion(PyObject *self, PyObject *args)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_DecodeASCII(qh_version, strlen(qh_version), NULL);
#else
    return PyString_FromString(qh_version);
#endif
}

/* Module methods */
static PyMethodDef Object3DQhullMethods[] = {
    {"delaunay", object3DDelaunay, METH_VARARGS, Object3DQhull_delaunay__doc__},
    {"voronoi",  object3DVoronoi, METH_VARARGS, Object3DQhull_delaunay__doc__},
    {"version",  getQhullVersion, METH_VARARGS},
    {NULL, NULL, 0, NULL} /* sentinel */
};

#if (REALfloat == 1)
#define MOD_NAME "Object3DQhullf"
#else
#define MOD_NAME "Object3DQhull"
#endif

#if PY_MAJOR_VERSION >= 3

static int Object3DQhull_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int Object3DQhull_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        MOD_NAME,
        Object3DQhull__doc__,
        sizeof(struct module_state),
        Object3DQhullMethods,
        NULL,
        Object3DQhull_traverse,
        Object3DQhull_clear,
        NULL
};

#define INITERROR return NULL

#if (REALfloat == 1)
PyObject *
PyInit_Object3DQhullf(void)
#else
PyObject *
PyInit_Object3DQhull(void)
#endif

#else
#define INITERROR return

#if (REALfloat == 1)
void
initObject3DQhullf(void)
#else
void
initObject3DQhull(void)
#endif
#endif
{
    struct module_state *st;
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3(MOD_NAME, Object3DQhullMethods, Object3DQhull__doc__);
#endif

    if (module == NULL)
        INITERROR;
    st = GETSTATE(module);

    st->error = PyErr_NewException("Object3DQhull.error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array()

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
