#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020-2021 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import sys
import numpy
import logging
_logger = logging.getLogger(__name__)

try:
    from PyMca5.PyMcaMath.mva import _cython_kmeans as _kmeans
    KMEANS = "_kmeans"
except:
    if _logger.getEffectiveLevel() == logging.DEBUG:
        raise
    else:
        _logger.warning("Cannot load built-in K-means.\n %s" % \
                         sys.exc_info()[1])
    KMEANS = None

try:
    from sklearn.cluster import KMeans
    KMEANS = "sklearn"
except:
    pass

if KMEANS:
    _logger.info("kmeans default to <%s>"  % KMEANS)
else:
    _logger.info("kmeans disabled")
    

def _labelCythonKMeans(x, k):
    labels, means, iterations, converged = _kmeans.kmeans(x, k)
    return {"labels": numpy.array(labels, dtype=numpy.int32, copy=False),
            "means": means,
            "iterations":iterations,
            "converged":converged}

def _labelMdp(x, k):
    from mdp.nodes import KMeansClassifier
    classifier = KMeansClassifier(k)
    for i in range(x.shape[0]):
        classifier.train(x[i:i+1])
    #classifier.train(x)
    labels = classifier.label(x)
    return {"labels": numpy.array(labels, dtype=numpy.int32, copy=False)}
     
def _labelScikitLearn(x, k):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k)
    km.fit(x)
    labels = km.labels_
    #labels = km.predict(x)
    converged = len(km.cluster_centers_) == len(labels)
    return {"labels": numpy.array(labels, dtype=numpy.int32, copy=False),
            "means": km.cluster_centers_,
            "iterations":km.n_iter_,
            "converged":converged}

def kmeans(x, k, method=None, normalize=True):
    """
    x is a 2D array [n_samples, n_features]
    k is the desired number of clusters
    """
    assert len(x.shape) == 2
    # collapse the information to deal with inf and NaNs
    raws = x.sum(axis=1, dtype=numpy.float64)
    good = numpy.isfinite(raws)
    finiteData = numpy.alltrue(good)
    data = numpy.ascontiguousarray(x[good])
    if normalize:
        datamin = data.min(axis=0)
        deltas = data.max(axis=0) - datamin 
        deltas[deltas < 1.0e-200] = 1
        data = (data - datamin) / deltas
    if method is None:
        method = KMEANS 
    if method == "mdp":
        result = _labelMdp(data, k)
    elif method == "sklearn":
        result = _labelScikitLearn(data, k)
    elif method.endswith("kmeans"):
        result = _labelCythonKMeans(data, k)
    elif "mdp" in sys.modules:
        result = _labelMdp(data, k)
    else:
        raise ValueError("Unknown clustering <%s>"  % method)
    if not finiteData:
        _logger.info("Data contains inf or NaNs")
        actualResult = -numpy.ones(raws.shape,dtype=numpy.int32)
        actualResult[good] = result["labels"]
        result["labels"] = actualResult
    return result

def label(*var, **kw):
    return kmeans(*var, **kw)["labels"]
