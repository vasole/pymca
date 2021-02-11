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
from PyMca5.PyMcaMath.fitting.Model import Model, ConcatModel


class SimpleModel(Model):
    """Model MCA data using a fixed list of peak positions and efficiencies"""

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
        self.sigma_to_fwhm = 2 * numpy.sqrt(2 * numpy.log(2))
        super(SimpleModel, self).__init__()

    def __str__(self):
        return "{}(npeaks={}, zero={}, gain={}, wzero={}, wgain={})".format(
            self.__class__, self.npeaks, self.zero, self.gain, self.wzero, self.wgain
        )

    @property
    def zero(self):
        return self.config["detector"]["zero"]

    @zero.setter
    def zero(self, value):
        self.config["detector"]["zero"] = value

    @property
    def gain(self):
        return self.config["detector"]["gain"]

    @gain.setter
    def gain(self, value):
        self.config["detector"]["gain"] = value

    @property
    def wzero(self):
        return self.config["detector"]["wzero"]

    @wzero.setter
    def wzero(self, value):
        self.config["detector"]["wzero"] = value

    @property
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
        return self.zero + self.gain * self.positions

    @property
    def areas(self):
        return self.efficiency * self.concentrations

    @property
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
    def nchannels(self):
        return self.xmax - self.xmin

    @property
    def npeaks(self):
        return len(self.concentrations)

    @property
    def _parameter_group_names(self):
        return ["zero", "gain", "wzero", "wgain", "concentrations"]

    @property
    def _linear_parameter_group_names(self):
        return ["concentrations"]

    def _iter_parameter_groups(self, linear_only=False):
        """
        :param bool linear_only:
        :yields (str, int): group name, nb. parameters in the group
        """
        if linear_only:
            names = self.linear_parameter_group_names
        else:
            names = self.parameter_group_names
        for name in names:
            if name == "zero":
                yield name, 1
            elif name == "gain":
                yield name, 1
            elif name == "wzero":
                yield name, 1
            elif name == "wgain":
                yield name, 1
            elif name == "concentrations":
                yield name, self.npeaks
            else:
                raise ValueError(name)

    def evaluate(self, xdata=None):
        """Evaluate model

        :param array xdata: length nxdata
        :returns array: nxdata
        """
        if xdata is None:
            xdata = self.xdata
        x = self.zero + self.gain * xdata
        p = list(zip(self.areas, self.positions, self.fwhms))
        return SpecfitFuns.agauss(p, x)

    def linear_derivatives(self, xdata=None):
        """Derivates to all linear parameters

        :param array xdata: length nxdata
        :returns array: nparams x nxdata
        """
        if xdata is None:
            xdata = self.xdata
        x = self.zero + self.gain * xdata
        it = zip(self.efficiency, self.positions, self.fwhms)
        return numpy.array([SpecfitFuns.agauss([a, p, w], x) for a, p, w in it])

    def derivative(self, param_idx, xdata=None):
        """Derivate to a specific parameter

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
            y = SpecfitFuns.agauss([a, p, w], x)
        else:
            fwhms = self.fwhms
            sigmas = fwhms / self.sigma_to_fwhm
            y = x * 0.0
            for p, a, w, s in zip(self.positions, self.areas, fwhms, sigmas):
                if name in ("zero", "gain"):
                    # Derivative to position
                    m = -(x - p) / s ** 2
                    # Derivative to position param
                    if name == "gain":
                        m *= xdata
                else:
                    # Derivative to FWHM
                    m = ((x - p) ** 2 / s ** 2 - 1) / (self.sigma_to_fwhm * s)
                    # Derivative to FWHM param
                    if name == "wgain":
                        m *= p
                y += m * SpecfitFuns.agauss([a, p, w], x)
        return y


class SimpleConcatModel(ConcatModel):
    def __init__(self, ndetectors=1):
        models = [SimpleModel() for i in range(ndetectors)]
        shared_attributes = ["concentrations", "positions"]
        super(SimpleConcatModel, self).__init__(
            models, shared_attributes=shared_attributes
        )
