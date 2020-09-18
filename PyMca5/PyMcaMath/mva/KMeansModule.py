#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import sys
import numpy
try:
    from sklearn.cluster import KMeans
    KMEANS = "sklearn"
except:
    try:
        from PyMca5.PyMcaMath.mva import _cython_kmeans as kmeans
        KMEANS = "_kmeans"
    except:
        KMEANS = False
    print(sys.exc_info())

def _labelCythonKMeans(x, k):
    labels, means, iterations, converged = kmeans.kmeans(x, k)
    return {"labels": labels,
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
    return {"labels": labels}
     
def _labelScikitLearn(x, k):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k)
    km.fit(x)
    labels = km.predict(x)
    return {"labels": labels}

def label(x, k, method=None, normalize=True):
    """
    x is a 2D array [n_observations, n_observables]
    k is the desired number of clusters
    """
    # TODO -> Deal with inf values and NaNs
    assert len(x.shape) == 2
    data = numpy.ascontiguousarray(x)
    if normalize:
        deltas = data.max(axis=1) - data.min(axis=1)
        deltas[deltas < 1.0e-200] = 1
        data = data / deltas[:, None]
    if method is None:
        method = KMEANS 
    if method == "mdp":
        labels = _labelMdp(data, k)["labels"]
    elif method == "sklearn":
        labels = _labelScikitLearn(data, k)["labels"]
    elif method.endswith("kmeans"):
        labels = _labelCythonKMeans(data, k)["labels"]
    elif "mdp" in sys.modules:
        labels = _labelMdp(data, k)["labels"]
    else:
        raise ValueError("Unknown clustering <%s>"  % method)
    return numpy.array(labels, dtype=numpy.int32, copy=False)
