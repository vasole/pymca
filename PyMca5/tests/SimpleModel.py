# /*##########################################################################
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
__author__ = "Wout De Nolf"
__contact__ = "wout.de_nolf@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"


import numpy
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaMath.fitting.model import parameter_group
from PyMca5.PyMcaMath.fitting.model import linear_parameter_group
from PyMca5.PyMcaMath.fitting.model import LeastSquaresFitModel
from PyMca5.PyMcaMath.fitting.model import LeastSquaresCombinedFitModel


class SimpleModel(LeastSquaresFitModel):
    """Model MCA data using a fixed list of peak positions and efficiencies"""

    SIGMA_TO_FWHM = 2 * numpy.sqrt(2 * numpy.log(2))

    def __init__(self):
        self.config = {
            "detector": {"zero": 0.0, "gain": 1.0, "wzero": 0.0, "wgain": 1.0},
            "matrix": {"positions": [], "concentrations": [], "efficiency": []},
            "fit": {"linear": False},
            "xmin": 0.0,
            "xmax": 1.0,
        }
        self.xdata_raw = None
        self.ydata_raw = None
        self.ystd_raw = None
        self.ybkg = 0
        super().__init__()

    def __str__(self):
        return "{}(npeaks={}, zero={}, gain={}, wzero={}, wgain={})".format(
            self.__class__, self.npeaks, self.zero, self.gain, self.wzero, self.wgain
        )

    @parameter_group
    def zero(self):
        return self.config["detector"]["zero"]

    @zero.setter
    def zero(self, value):
        self.config["detector"]["zero"] = value

    @parameter_group
    def gain(self):
        return self.config["detector"]["gain"]

    @gain.setter
    def gain(self, value):
        self.config["detector"]["gain"] = value

    @parameter_group
    def wzero(self):
        return self.config["detector"]["wzero"]

    @wzero.setter
    def wzero(self, value):
        self.config["detector"]["wzero"] = value

    @parameter_group
    def wgain(self):
        return self.config["detector"]["wgain"]

    @wgain.setter
    def wgain(self, value):
        self.config["detector"]["wgain"] = value

    @property
    def efficiency(self):
        return self.config["matrix"]["efficiency"]

    @efficiency.setter
    def efficiency(self, value):
        arr = self.config["matrix"]["efficiency"]
        self.config["matrix"]["efficiency"] = value

    @property
    def positions(self):
        return self.config["matrix"]["positions"]

    @positions.setter
    def positions(self, value):
        self.config["matrix"]["positions"] = value

    @property
    def fwhms(self):
        return self.wzero + self.wgain * self.positions

    @property
    def areas(self):
        return self.efficiency * self.concentrations

    @linear_parameter_group
    def concentrations(self):
        return self.config["matrix"]["concentrations"]

    @concentrations.setter
    def concentrations(self, value):
        self.config["matrix"]["concentrations"] = value

    @property
    def linear(self):
        return self.config["fit"]["linear"]

    @linear.setter
    def linear(self, value):
        self.config["fit"]["linear"] = value

    @property
    def idx_channels(self):
        return slice(self.xmin, self.xmax)

    @property
    def xdata(self):
        if self.xdata_raw is None:
            return None
        else:
            return self.xdata_raw[self.idx_channels]

    @xdata.setter
    def xdata(self, values):
        self.xdata_raw[self.idx_channels] = values

    @property
    def xenergy(self):
        return self.zero + self.gain * self.xdata

    def _y_full_to_fit(self, y, xdata=None):
        return y - self.ybkg

    def _y_fit_to_full(self, y, xdata=None):
        return y + self.ybkg

    @property
    def ydata(self):
        if self.ydata_raw is None:
            return None
        else:
            return self.ydata_raw[self.idx_channels]

    @ydata.setter
    def ydata(self, values):
        self.ydata_raw[self.idx_channels] = values

    @property
    def ystd(self):
        if self.ystd_raw is None:
            return None
        else:
            return self.ystd_raw[self.idx_channels]

    @ystd.setter
    def ystd(self, values):
        self.ystd_raw[self.idx_channels] = values

    @property
    def ndata(self):
        return self.xmax - self.xmin

    @property
    def npeaks(self):
        return len(self.concentrations)

    def evaluate_fitmodel(self, xdata=None):
        """Evaluate model

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        x = self.zero + self.gain * xdata
        p = list(zip(self.areas, self.positions, self.fwhms))
        return SpecfitFuns.agauss(p, x)

    def derivative_fitmodel(self, param_idx, xdata=None):
        """Derivate to a specific parameter_group

        :param int param_idx:
        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        x = self.zero + self.gain * xdata

        name, i = self._parameter_name_from_index(param_idx)
        if name == "concentrations":
            p = self.positions[i]
            a = self.efficiency[i]
            w = self.wzero + self.wgain * p
            return SpecfitFuns.agauss([a, p, w], x)

        fwhms = self.fwhms
        sigmas = fwhms / self.SIGMA_TO_FWHM
        y = x * 0.0
        for p, a, w, s in zip(self.positions, self.areas, fwhms, sigmas):
            if name in ("zero", "gain"):
                # Derivative to position
                m = -(x - p) / s ** 2
                # Derivative to position param
                if name == "gain":
                    m *= xdata
            elif name in ("wzero", "wgain"):
                # Derivative to FWHM
                m = ((x - p) ** 2 / s ** 2 - 1) / (self.SIGMA_TO_FWHM * s)
                # Derivative to FWHM param
                if name == "wgain":
                    m *= p
            else:
                raise ValueError(name)
            y += m * SpecfitFuns.agauss([a, p, w], x)
        return y


class SimpleCombinedModel(LeastSquaresCombinedFitModel):
    def __init__(self, ndetectors=1):
        models = {f"detector{i}":SimpleModel() for i in range(ndetectors)}
        super().__init__(models)
        self._enable_property_link("concentrations", "positions")
