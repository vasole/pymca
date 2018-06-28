__contact__ = "jerome.kieffer@esrf.eu"
__license__ = """
Copyright (c) J. Kieffer, European Synchrotron Radiation Facility

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""
Module Sift for calculating SIFT keypoint using PyOpenCL
"""
version = "0.2.0"
import os
sift_home = os.path.dirname(os.path.abspath(__file__))
import sys, logging
logging.basicConfig()
from .plan import SiftPlan
from .match import MatchPlan
from .alignment import LinearAlign


_logger = logging.getLogger(__name__)
_logger.warning("The sift module in PyMca is deprecated. "
                "You should import sift from the silx library.")
