#!/usr/bin/python
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
import pyopencl,pyopencl.array
import numpy
ctx = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
x, y, z = numpy.ogrid[-10:10:0.05, -10:10:0.05, -10:10:0.05]
r=numpy.sqrt(x*x+y*y+z*z)
data = ((x * x - y * y + z * z) * numpy.exp(-r)).astype("float32")
gpu_vol = pyopencl.image_from_array(ctx, data, 1)
shape = (500, 500)
img = numpy.empty(shape,dtype=numpy.float32)
gpu_img = pyopencl.array.empty(queue, shape, numpy.float32)
prg = open("interpolation.cl").read()
sampler = pyopencl.Sampler(ctx,
                           True, # normalized coordinates
                           pyopencl.addressing_mode.CLAMP_TO_EDGE,
                           pyopencl.filter_mode.LINEAR)

prg = pyopencl.Program(ctx, prg).build()
n = pyopencl.array.to_device(queue, numpy.array([1, 1, 1], dtype=numpy.float32))
c = pyopencl.array.to_device(queue, numpy.array([0.5, 0.5, 0.5], dtype=numpy.float32))
prg.interpolate(queue, (512, 512), (16, 16), gpu_vol, sampler, gpu_img.data,
                numpy.int32(shape[1]), numpy.int32(shape[1]), c.data, n.data)
img = gpu_img.get()


#timing:
evt = []
evt.append(pyopencl.enqueue_copy(queue, n.data, (2.0*numpy.random.random(3)-1).astype(numpy.float32)))
evt.append(pyopencl.enqueue_copy(queue, c.data, numpy.random.random(3).astype(numpy.float32)))
evt.append(prg.interpolate(queue, (512, 512), (16, 16), gpu_vol, sampler, gpu_img.data,
                numpy.int32(shape[1]), numpy.int32(shape[0]), c.data, n.data))
evt.append(pyopencl.enqueue_copy(queue, img, gpu_img.data))
print("Timings: %.3fms %.3fms %.3fms %.3fms total: %.3fms" % (1e-6 * (evt[0].profile.end - evt[0].profile.start),
                                1e-6 * (evt[1].profile.end - evt[1].profile.start),
                                1e-6 * (evt[2].profile.end - evt[2].profile.start),
                                1e-6 * (evt[3].profile.end - evt[3].profile.start),
                                1e-6 * (evt[-1].profile.end - evt[0].profile.start)))


from pylab import *
imshow(img)
show()
