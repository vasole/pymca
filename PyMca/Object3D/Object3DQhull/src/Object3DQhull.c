#include <Python.h>
#include <stdlib.h>
#include <stdio.h>

#include <./numpy/arrayobject.h>

#include "libqhull.h"
#include "qset.h"		/* for FOREACHneighbor_() */
#include "poly.h"		/* for qh_vertexneighbors() */


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


/* static variables */
static PyObject *Object3DQhullError;

/* Function declarations */
static PyObject *object3DDelaunay(PyObject *dummy, PyObject *args);
static PyObject *object3DVoronoi(PyObject *dummy, PyObject *args);
static void qhullResultFailure(int);
static PyObject *getQhullVersion(PyObject *dummy, PyObject *args);


static PyObject *object3DDelaunay(PyObject *self, PyObject *args)
{
	/* input parameters */
	PyObject	*input1, *input3=NULL;
	const char      *input2 = NULL;	

	/* local variables */
	PyArrayObject	*pointArray, *inner_pointArray=NULL;
	PyArrayObject	*result, *inner_result=NULL ;

	coordT	*points;	/* Qhull */
	int		dimension;	/* Qhull */
	int		nPoints;	/* Qhull */
	int		inner_nPoints = 0;	/* Qhull */

	int		qhullResult;		/* Qhull exit code, 0 means no error */
	boolT ismalloc = False;		/* True if Qhull should free points in
								   qh_freeqhull() or reallocation */
	//char cQhullDefaultFlags[] = "qhull d Qbb Qt"; /* Qhull flags (see doc)*/
#if (REALfloat == 1)
    char cQhullDefaultFlags[] = "qhull d Qbb QJ Qc Po"; /* Qhull flags (see doc) Po is to ignore precision errors*/
#else
	char cQhullDefaultFlags[] = "qhull d Qbb QJ Qc"; /* Qhull flags (see doc)*/
#endif
    char *cQhullFlags;
	
	int			nFacets = 0;
	npy_intp	outDimensions[3];
	facetT *facet;		/* needed by FORALLfacets */
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




    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "O|zO", &input1, &input2, &input3 ))
	{
		PyErr_SetString(Object3DQhullError, "Unable to parse arguments");
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
	    PyErr_SetString(Object3DQhullError, "third argument if given must be  a nrows x X array");
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
	    PyErr_SetString(Object3DQhullError, "third argument if given must be  a nrows x X array");
	    return NULL;
	  }
	}
#endif

 
    if (pointArray == NULL)
	{
	    PyErr_SetString(Object3DQhullError, "First argument is not a nrows x X array");
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
    dimension = pointArray->dimensions[1];
    
    /* number of points for Qhull */
    nPoints = pointArray->dimensions[0];
    
    
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
		qhullResultFailure(qhullResult);
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
    if (0)	// As triangles
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
	    PyErr_SetString(Object3DQhullError, "Error allocating output memory");
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
	  FOREACHvertex_(facet->vertices)	{
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

		if( input3 ) {
		  inner_nPoints = inner_pointArray->dimensions[0];

		  outDimensions[0] = inner_nPoints;
		  outDimensions[1] = dimension+1;
		  inner_result = (PyArrayObject *)
		    PyArray_SimpleNew(2, outDimensions, PyArray_UINT32);
		  if (inner_result == NULL)
		    {
		      qh_freeqhull(qh_ALL);
		      Py_DECREF (pointArray);
		      Py_DECREF (inner_pointArray);
		      PyErr_SetString(Object3DQhullError, "Error allocating output memory for inner points facets");
		      return NULL;
		    }
		}

		if (result == NULL)
		{
			qh_freeqhull(qh_ALL);
			Py_DECREF (pointArray);
			{
			  if(inner_pointArray){  Py_DECREF (inner_pointArray) ;} 
			}
			PyErr_SetString(Object3DQhullError, "Error allocating output memory");
			return NULL;
		}

		uintP = (unsigned int *) result->data;
		FORALLfacets {
			if (facet->upperdelaunay)
				continue;
			FOREACHvertex_(facet->vertices)	{
					*uintP =  qh_pointid(vertex->point);
					++uintP;
			}
		}
		if(input3) {
		  uintP = (unsigned int *) inner_result->data;
#if (REALfloat == 1) 
		  p = (float *) inner_pointArray->data;
#else
		  p = (double *) inner_pointArray->data;
#endif
		  for (i=0; i< inner_nPoints; i++)
		    {
		      for(j=0; j<dimension; j++) {
			point[j] = *( p++)     ;
		      }
		      
		      qh_setdelaunay(  dimension+1 , 1, point);
		      
		      facet = qh_findbestfacet(point, qh_ALL, &bestdist, &isoutside);


		      if (facet && !facet->upperdelaunay && facet->simplicial) {
			FOREACHvertex_(facet->vertices)	{
			  *uintP =  qh_pointid(vertex->point);
			  ++uintP;
			}			
		      } else {
			{ qh_freeqhull(qh_ALL);} 
			Py_DECREF (pointArray);
			if(pointArray) {  Py_DECREF (inner_pointArray);} 
			PyErr_SetString(Object3DQhullError, "Error allocating output memory");
			return NULL;
		      }
		    }
		}
	}


	/* Free the memory allocated by Qhull */
	qh_freeqhull(qh_ALL);
	
	Py_DECREF (pointArray);
	
	if( input3==NULL) {
	  return PyArray_Return(result);
	} else {
	  return  Py_BuildValue("NN", result, inner_result );
	}
}

static PyObject *object3DVoronoi(PyObject *self, PyObject *args)
{
	/* input parameters */
	PyObject	*input1, *input3=NULL;
	const char      *input2 = NULL;	

	/* local variables */
	PyArrayObject	*pointArray, *inner_pointArray=NULL;
	PyArrayObject	*result, *inner_result=NULL ;

	coordT	*points;	/* Qhull */
	int		dimension;	/* Qhull */
	int		nPoints;	/* Qhull */
	int		inner_nPoints = 0;	/* Qhull */

	int		qhullResult;		/* Qhull exit code, 0 means no error */
	boolT ismalloc = False;		/* True if Qhull should free points in
								   qh_freeqhull() or reallocation */
	//char cQhullDefaultFlags[] = "qhull d Qbb Qt"; /* Qhull flags (see doc)*/
#if (REALfloat == 1)
    char cQhullDefaultFlags[] = "qhull v p"; /* Qhull flags (see doc) Po is to ignore precision errors*/
#else
	char cQhullDefaultFlags[] = "qhull v p"; /* Qhull flags (see doc)*/
#endif
    char *cQhullFlags;
	
	int			nFacets = 0;
	npy_intp	outDimensions[2];
	facetT *facet;		/* needed by FORALLfacets */
	vertexT *vertex, **vertexp;
	setT *vertices;
	pointT *center;

  int numcenters, format=0, numvertices= 0, numneighbors, numinf, vid=1, vertex_i, vertex_n;
  facetT *neighbor, **neighborp;
  boolT islower;
  unsigned int numfacets;


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




    /* ------------- statements ---------------*/
    if (!PyArg_ParseTuple(args, "O|zO", &input1, &input2, &input3 ))
	{
		PyErr_SetString(Object3DQhullError, "Unable to parse arguments");
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
	    PyErr_SetString(Object3DQhullError, "third argument if given must be  a nrows x X array");
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
	    PyErr_SetString(Object3DQhullError, "third argument if given must be  a nrows x X array");
	    return NULL;
	  }
	}
#endif

 
    if (pointArray == NULL)
	{
	    PyErr_SetString(Object3DQhullError, "First argument is not a nrows x X array");
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
    dimension = pointArray->dimensions[1];
    
    /* number of points for Qhull */
    nPoints = pointArray->dimensions[0];
    
    
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
		qhullResultFailure(qhullResult);
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
	printf("output dimensions = %d, %d\n",outDimensions[0], outDimensions[1]); 
	result = (PyArrayObject *)
	  PyArray_SimpleNew(2, outDimensions, PyArray_DOUBLE);
	if (result == NULL)
	  {
	    qh_freeqhull(qh_ALL);
	    Py_DECREF (pointArray);
		if(input3) {
			Py_DECREF (inner_pointArray);
		}
	    PyErr_SetString(Object3DQhullError, "Error allocating output memory");
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
qhullResultFailure(int qhull_exitcode)
{
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
		PyErr_SetString(PyExc_StandardError,
				"qhull internal error");
		break;
	}
}

static PyObject *getQhullVersion(PyObject *self, PyObject *args)
{
    return PyString_FromString(qh_version);
}

/* Module methods */
static PyMethodDef Object3DQhullMethods[] = {
	{"delaunay", object3DDelaunay, METH_VARARGS, Object3DQhull_delaunay__doc__},
	{"voronoi",  object3DVoronoi, METH_VARARGS, Object3DQhull_delaunay__doc__},
    {"version",  getQhullVersion, METH_VARARGS},
	{NULL, NULL, 0, NULL} /* sentinel */
};


/* Initialise the module. */
#if (REALfloat == 1)
PyMODINIT_FUNC
initObject3DQhullf(void)
{
	PyObject	*m, *d;
	/* Create the module and add the functions */
	m = Py_InitModule3("Object3DQhullf", Object3DQhullMethods, Object3DQhull__doc__);
	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);

	import_array()
	Object3DQhullError = PyErr_NewException("Object3DQhullf.error", NULL, NULL);
	PyDict_SetItemString(d, "error", Object3DQhullError);
}
#else
PyMODINIT_FUNC
initObject3DQhull(void)
{
	PyObject	*m, *d;
	/* Create the module and add the functions */
	m = Py_InitModule3("Object3DQhull", Object3DQhullMethods, Object3DQhull__doc__);
	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);

	import_array()
	Object3DQhullError = PyErr_NewException("Object3DQhull.error", NULL, NULL);
	PyDict_SetItemString(d, "error", Object3DQhullError);
}
#endif
