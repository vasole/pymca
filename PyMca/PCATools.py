#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
import numpy
import numpy.linalg
try:
    import numpy.core._dotblas as dotblas
except ImportError:
    print("WARNING: Not using BLAS, PCA calculation will be slower")
    dotblas = numpy

DEBUG = 0
def getCovarianceMatrix(stack,
                        index=-1,
                        binning=None,
                        dtype='float64',
                        force=True,
                        center=True,
                        weights=None):
    #the 1D mask should correspond to the values, before or after sampling?
    #it could be handled as weigths to be applied to the spectra. That would
    #allow two uses, as mask and as weights, at the cost of a multiplication.
    #recover the actual data to work with
    if hasattr(stack, "info") and hasattr(stack, "data"):
        #we are dealing with a PyMca data object
        data = stack.data
    else:
        data = stack
        
    oldShape = data.shape
    if index not in [0, -1, len(oldShape)-1]:
        data = None
        raise IndexError("1D index must be one of 0, -1 or %d" % len(oldShape))

    if index < 0:
        actualIndex = len(oldShape) + index
    else:
        actualIndex = index

    print "actual Index", actualIndex
    #the number of spatial pixels
    nPixels = 1
    for i in range(len(oldShape)):
        if i != actualIndex:
            nPixels *= oldShape[i]

    #the starting number of channels or of images
    N=oldShape[index]

    #our binning (better said sampling) is spectral, in order not to affect the spatial resolution
    if binning is None:
        binning = 1

    if weights is None:
        weights = numpy.ones(N, numpy.float)

    nChannels = int(N/binning)
    cleanWeights = weights[::binning]
    print "USING binning = ", binning

    #end of checking part
    eigenvectorLength = nChannels

    if (not force)and isinstance(data, numpy.ndarray):
        #make a direct calculation (memory cosuming)
        #take a view to the data
        dataView = data[:]
        if index in [0]:
            #reshape the view to allow the matrix multiplication
            dataView.shape = -1, nPixels
            cleanWeights.shape = -1, 1
            dataView = dataView[::binning] * cleanWeights
            sumSpectrum = dataView.sum(axis=1, dtype=numpy.float64)
            #and return the standard covariance matrix as a matrix product
            covMatrix = dotblas.dot(dataView, dataView.T)/float(nPixels-1)
        else:
            #the last index
            dataView.shape = nPixels, -1
            cleanWeights.shape = 1, -1
            dataView = dataView[:,::binning] * cleanWeights
            sumSpectrum = dataView.sum(axis=0, dtype=numpy.float64)
            #and return the standard covariance matrix as a matrix product
            covMatrix = dotblas.dot(dataView.T, dataView)/ float(nPixels-1)
        if center:
            averageMatrix = numpy.outer(sumSpectrum, sumSpectrum)/(nPixels * (nPixels - 1))
            covMatrix -= averageMatrix
            averageMatrix = None
        return covMatrix, sumSpectrum, nPixels

    #we are dealing with dynamically loaded data
    print("DINAMICALLY LOADED DATA")
    #create the needed storage space for the covariance matrix
    try:
        covMatrix = numpy.zeros((eigenvectorLength, eigenvectorLength), numpy.float64)
        sumSpectrum = numpy.zeros((eigenvectorLength,), numpy.float64)
        nSpectra = nPixels
    except:
        #make sure no reference to the original input data is kept
        cleanWeights = None
        covMatrix = None
        averageMatrix = None
        data = None
        raise

    if index in [0]:
        #divider is used to decide the fraction of images to keep in memory
        #in order to limit file access on dynamically loaded data.
        #Since two chunks of the same size are used, the amount of memory
        #needed is twice the data size divided by the divider.
        #For instance, divider = 10 implies the data to be read 5.5 times
        #from disk while having a memory footprint of about one fifth of
        #the dataset size.
        step = 0
        divider = 10
        while step < 1:
            step = int(oldShape[index]/divider)
            divider -= 2
            if divider <= 0:
                step = oldShape[index]
                break
        if DEBUG:
            print("Reading chunks of %d images" % step)
        nImagesRead = 0
        if (binning == 1) and oldShape[index] >= step:
            chunk1 = numpy.zeros((step, nPixels), numpy.float64)
            chunk2 = numpy.zeros((nPixels, step), numpy.float64)
            i = 0
            while i < N:
                iToRead = min(step, N - i)
                
                #get step images for the first chunk
                if 0:
                    for k in range(0, iToRead):
                        chunk1[k] = data[(i+k):(i+k+1)].reshape(-1)
                else:
                    chunk1[0:iToRead] = data[i:i+iToRead].reshape(iToRead,-1)
                sumSpectrum[i:(i+iToRead)] = chunk1[0:iToRead].sum(axis=1)
                if center:
                    average = sumSpectrum[i:(i+iToRead)]/nPixels
                    average.shape = iToRead, 1
                    chunk1[0:iToRead] -= average
                nImagesRead += iToRead
                j = 0
                while j <= i:
                    #get step images for the second chunk
                    if j == i:
                        jToRead = iToRead
                        if 0:
                            for k in range(0, jToRead):
                                chunk2[:,k] = chunk1[k]
                        else:
                            chunk2[:,0:jToRead] = chunk1[0:jToRead,:].T
                    else:
                        #get step images for the second chunk
                        jToRead = min(step, nChannels-j)
                        for k in range(0, jToRead):
                            chunk2[:,k] = data[(j+k):(j+k+1)].reshape(1,-1)
                        nImagesRead += jToRead
                        if center:
                            average = chunk2[:,0:jToRead].sum(axis=0)/nPixels
                            average.shape = 1, jToRead
                            chunk2 -= average

                    #dot product
                    if (iToRead != step) or (jToRead != step):
                        covMatrix[i:(i+iToRead), j:(j+jToRead)] =\
                                         dotblas.dot(chunk1[:iToRead,:nPixels],
                                                     chunk2[:nPixels,:jToRead])
                    else:
                        covMatrix[i:(i+iToRead), j:(j+jToRead)] =\
                                                 dotblas.dot(chunk1, chunk2)

                    if i != j:
                        covMatrix[j:(j+jToRead), i:(i+iToRead)] =\
                                    covMatrix[i:(i+iToRead), j:(j+jToRead)].T

                    #increment j
                    j+= jToRead                
                i += iToRead
            chunk1 = None
            chunk2 = None
            if DEBUG:
                print("totalImages Read = ", nImagesRead)
        elif (binning > 1) and (oldShape[index] >= step):
            chunk1 = numpy.zeros((step, nPixels), numpy.float64)
            chunk2 = numpy.zeros((nPixels, step), numpy.float64)
            #one by one reading till we fill the chunks
            imagesToRead = numpy.arange(0,oldShape[index],binning)
            i = int(imagesToRead[weights>0][0])
            spectrumIndex = 0
            nImagesRead = 0
            while i < N:
                #fill chunk1
                jj = 0
                for iToRead in range(0, int(min(step*binning, N - i)),binning):
                    chunk1[jj] = data[i+iToRead].reshape(1,-1) * weights[i+iToRead]
                    jj += 1
                sumSpectrum[spectrumIndex:spectrumIndex+jj] = chunk1[0:jj].sum(axis=1)
                if center:
                    average = sumSpectrum[spectrumIndex:spectrumIndex+jj]/nPixels
                    average.shape = jj, 1
                    chunk1[0:jj] -= average
                nImagesRead += jj
                iToRead = jj
                j = 0
                while j <= i:
                    #get step images for the second chunk
                    if j == i:
                        jToRead = iToRead
                        chunk2[:,0:jToRead] = chunk1[0:jToRead,:].T
                    else:
                        #get step images for the second chunk
                        jj = 0
                        for jToRead in range(0, int(min(step*binning, N - j)),binning):
                            chunk2[:,jj] = data[j+jToRead].reshape(1,-1) * weights[j+jToRead]
                            jj += 1
                        nImagesRead += jj
                        if center:
                            average = chunk2[:,0:jj].sum(axis=0)/nPixels
                            average.shape = 1, jj
                            chunk2 -= average
                        jToRead = jj
                    #dot product
                    if (iToRead != step) or (jToRead != step):
                        covMatrix[i:(i+iToRead), j:(j+jToRead)] =\
                                         dotblas.dot(chunk1[:iToRead,:nPixels],
                                                     chunk2[:nPixels,:jToRead])
                    else:
                        covMatrix[i:(i+iToRead), j:(j+jToRead)] =\
                                                 dotblas.dot(chunk1, chunk2)

                    if i != j:
                        covMatrix[j:(j+jToRead), i:(i+iToRead)] =\
                                    covMatrix[i:(i+iToRead), j:(j+jToRead)].T

                    #increment j
                    j+= jToRead * step                
                i += iToRead * step
            chunk1 = None
            chunk2 = None
        else:
            raise ValueError("Unhandled case")
            
        #should one divide by N or by N-1 ??
        covMatrix /= nChannels
    else:
        #the data are already arranged as (nPixels, nChannels) and we
        #basically have to return data.T * data to get back the covariance
        #matrix as (nChannels, nChannels)
        #if someone had the bad idea to store the data in HDF5 with a chunk
        #size based on the pixels and not on the spectra a loop based on
        #reading spectrum per spectrum can be very slow
        step = 0
        divider = 10
        while step < 1:
            step = int(nPixels/divider)
            #divider = int(divider/2)
            divider -= 1
            if divider <= 0:
                step = nPixels
                break
        step = nPixels
        if DEBUG:
            print("Reading chunks of %d spectra" % step)

        cleanWeights.shape = 1, -1
        if len(data.shape) == 2:
            tmpData = numpy.zeros((step, nChannels), numpy.float64)
            k = 0
            while k < nPixels:
                kToRead = min(step, nPixels-k)
                tmpData[0:kToRead] = data[k:k+kToRead,::binning]
                a = tmpData[0:kToRead] * cleanWeights
                sumSpectrum += a.sum(axis=0)
                covMatrix += dotblas.dot(a.T, a)
                a = None
                k += kToRead
            tmpData = None
        elif len(data.shape) == 3:
            if oldShape[0] == 1:
                #close to the previous case
                tmpData = numpy.zeros((step, nChannels), numpy.float64)
                for i in range(oldShape[0]):
                    k = 0
                    while k < oldShape[1]:
                        kToRead = min(step, oldShape[1]-k)
                        tmpData[0:kToRead] = data[i,k:k+kToRead,::binning] * cleanWeights
                        a = tmpData[0:kToRead]
                        sumSpectrum += a.sum(axis=0)
                        covMatrix += dotblas.dot(a.T, a)
                        a = None
                        k += kToRead
                tmpData = None
            elif oldShape[1] == 1:
                #almost identical to the previous case
                tmpData = numpy.zeros((step, nChannels), numpy.float64)
                for i in range(oldShape[1]):
                    k = 0
                    while k < oldShape[0]:
                        kToRead = min(step, oldShape[0]-k)
                        tmpData[0:kToRead] = data[k:k+kToRead,i,::binning] * cleanWeights
                        a = tmpData[0:kToRead]
                        sumSpectrum +=  a.sum(axis=0)
                        covMatrix += dotblas.dot(a.T, a)
                        a = None
                        k += kToRead
                tmpData = None
            elif oldShape[0] < 21:
                if step > oldShape[1]:
                    step = oldShape[1]
                tmpData = numpy.zeros((step, nChannels), numpy.float64)
                for i in range(oldShape[0]):
                    k = 0
                    while k < oldShape[1]:
                        kToRead = min(step, oldShape[1] - k)
                        tmpData[0:kToRead] = data[i,k:k+kToRead,::binning] * cleanWeights
                        a = tmpData[0:kToRead]
                        sumSpectrum +=  a.sum(axis=0)
                        covMatrix += dotblas.dot(a.T, a)
                        a = None
                        k += kToRead
                tmpData = None
            else:
                #I should choose the sizes in terms of the size
                #of the dataset
                if oldShape[0] < 41:
                    #divide by 10
                    deltaRow = 4
                elif oldShape[0] < 101:
                    #divide by 10
                    deltaRow = 10
                else:
                    #take pieces of one tenth
                    deltaRow = int(oldShape[0]/10)
                deltaCol = oldShape[1]
                tmpData = numpy.zeros((deltaRow, deltaCol, nChannels),
                                      numpy.float64)
                i = 0
                while i < oldShape[0]:
                    iToRead = min(deltaRow, oldShape[0] - i)
                    kToRead = iToRead * oldShape[1]
                    tmpData[:iToRead] = data[i:(i+iToRead),:,::binning]
                    a = tmpData[:iToRead]
                    a.shape = kToRead, nChannels
                    a *= cleanWeights
                    if 0:
                        #weight each spectrum
                        a /= (a.sum(axis=1).reshape(-1, 1))
                    sumSpectrum +=  a.sum(axis=0)
                    covMatrix += dotblas.dot(a.T, a)
                    a = None
                    i += iToRead
        #should one divide by N or by N-1 ??
        covMatrix /= (nPixels-1)
        if center:
            #the n-1 appears again here
            averageMatrix = numpy.outer(sumSpectrum, sumSpectrum)/(nPixels * (nPixels - 1))
            covMatrix -= averageMatrix
            averageMatrix = None
    return covMatrix, sumSpectrum/nPixels, nPixels
        

def numpyPCA(stack, index=-1, ncomponents=10, binning=None, **kw):
    #recover the actual data to work with
    if hasattr(stack, "info") and hasattr(stack, "data"):
        #we are dealing with a PyMca data object
        data = stack.data
    else:
        data = stack
        
    oldShape = data.shape
    if index not in [0, -1, len(oldShape)-1]:
        data = None
        raise IndexError("1D index must be one of 0, -1 or %d, got %d" % (len(oldShape)-1, index))

    if index < 0:
        actualIndex = len(oldShape) + index
    else:
        actualIndex = index

    #the number of spatial pixels
    nPixels = 1
    for i in range(len(oldShape)):
        if i != actualIndex:
            nPixels *= oldShape[i]

    #the number of channels
    nChannels = oldShape[actualIndex]
    if binning is None:
        binning = 1

    N = int(nChannels/binning)

    cov, sumSpectrum, calculatedPixels = getCovarianceMatrix(stack,
                                                             index=index,
                                                             binning=binning,
                                                             force=False,
                                                             center=True)

    #the total variance is the sum of the elements of the diagonal
    totalVariance = numpy.diag(cov)
    print("Total Variance = ", totalVariance.sum())

    #option to normalize to unit standard deviation
    normalizeToUnitStandardDeviation = False
    if normalizeToUnitStandardDeviation:
        for i in range(cov.shape[0]):
            if totalVariance[i] != 0:
                cov[i] = cov[i]/totalVariance[i]

    if DEBUG:
        import time
        t0 = time.time()
    evalues, evectors = numpy.linalg.eigh(cov)
    if DEBUG:
        print("Eig elapsed = ", time.time() - t0)
    cov = None

    dtype = numpy.float32
    images = numpy.zeros((ncomponents, nPixels), dtype)
    eigenvectors = numpy.zeros((ncomponents, N), dtype)
    eigenvalues = numpy.zeros((ncomponents,), type)
    #sort eigenvalues
    if 1:
        a = [(evalues[i], i) for i in range(len(evalues))]
        a.sort()
        a.reverse()
        for i0 in range(ncomponents):
            i = a[i0][1]
            eigenvalues[i0] = evalues[i]
            if normalizeToUnitStandardDeviation:
                print("Explained variance = ", i, evalues[i]/totalVariance) 
            else:
                print("Explained variance = ", i, evalues[i]/totalVariance.shape[0])
            eigenvectors[i0,:] = evectors[:,i]
    else:
        idx = numpy.argsort(evalues)
        eigenvalues[:]  = evalues[idx]
        eigenvectors[:,:] = evectors[:,idx].T 

    #calculate the projections
    if actualIndex in [0]:
        for i in range(oldShape[actualIndex]):
            tmpData = data[i].reshape(1,-1)
            for j in range(ncomponents):
                images[j:j+1,:] += tmpData * eigenvectors[j, i]
        if len(oldShape) == 3:
            #reshape the images
            images.shape = ncomponents, oldShape[1], oldShape[2]
    else:
        #array of spectra
        if len(oldShape) == 2:
            for i in range(nPixels):
                #print i
                tmpData = data[i, :]
                tmpData.shape = 1, nChannels
                tmpData = tmpData[:,::binning]
                for j in range(ncomponents):
                    images[j, i] = numpy.dot(tmpData, eigenvectors[j])
            #reshape the images
            images.shape = ncomponents, nPixels
        elif len(oldShape) == 3:
            i = 0
            for r in range(oldShape[0]):
                for c in range(oldShape[1]):
                    #print i
                    tmpData = data[r, c, :]
                    tmpData.shape = 1, nChannels
                    tmpData = tmpData[:,::binning]
                    for j in range(ncomponents):
                        images[j, i] = numpy.dot(tmpData, eigenvectors[j])
                    i += 1
            #reshape the images
            images.shape = ncomponents, oldShape[0], oldShape[1]
    
    return images, eigenvalues, eigenvectors

def test():
    x = numpy.array([[0.0, 2.0,  3.0],
                     [3.0, 0.0, -1.0],
                     [4.0, -4.0, 4.0],
                     [4.0, 4.0,  4.0]])
    shape0 = x.shape
    print("x:")
    print(x)
    print("Numpy covariance matrix. It uses (n-1)")
    print(numpy.cov(x.T))
    avg = x.sum(axis=0).reshape(-1,1)/x.shape[0]
    print("Average = ", avg)
    print("OPERATION")
    print(numpy.dot((x.T-avg), (x.T-avg).T)/(x.shape[0]-1))

    print("PCATools.getCovarianceMatrix(x, force=True)")
    x.shape = 1, shape0[0], shape0[1]
    pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x, force=True)
    print("PyMca covariance matrix. It uses (n-1)")
    print(pymcaCov)
    print("Average = ", pymcaAvg)

    print("PCATools.getCovarianceMatrix(x)")
    x.shape = shape0[0], 1,  shape0[1]
    pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x)
    print("PyMca covariance matrix. It uses (n-1)")
    print(pymcaCov)
    print("Average = ", pymcaAvg)

    print("MDP")
    import mdp
    pca = mdp.nodes.PCANode(dtype=numpy.float)
    x.shape = shape0
    pca.train(x)
    pca._stop_training(debug=True)
    print("MDP covariance matrix. It uses (n-1)")
    print(pca.cov_mtx)
    print("Average = ", pca.avg)

    print("test as Images")
    stack = numpy.zeros((shape0[-1], shape0[0],1), numpy.float)
    for i in range(stack.shape[0]):
        stack[i,:,0] = x[:,i]
    #print stack[0]
    #print "x = ", x
    x = stack
    print("PCATools.getCovarianceMatrix(x)")
    pymcaCov, pymcaAvg, nData = getCovarianceMatrix(x, index=0, force=True)
    print("PyMca covariance matrix. It uses (n-1)")
    print(pymcaCov)
    print("Average = ", pymcaAvg)


if __name__ == "__main__":
    import sys
    test()
    sys.exit(0)
    from PyMca import EDFStack
    from PyMca import EdfFile
    import os
    import sys
    import time
    #inputfile = ".\PierreSue\CH1777\G4-Sb\G4_mca_0012_0000_0000.edf"
    inputfile = "D:\DATA\COTTE\ch09\ch09__mca_0005_0000_0000.edf"    
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
        print(inputfile)
    elif os.path.exists(inputfile):
        print("Using a default test case")
    else:
        print("Usage:")
        print("python PCAModule.py indexed_edf_stack")
        sys.exit(0)
    if 1:
        if 0:
            stack = EDFStack.EDFStack(inputfile, imagestack=True, dtype=numpy.float64)
            nImages, r, c = stack.data.shape
            stack.data.shape = nImages, r * c
            t0 = time.time()
            nPixels0 = r * c
            sumSpectrum0 = stack.data.sum(axis=0, dtype=numpy.float64)
            covMatrix0 = dotblas.dot(stack.data, stack.data.T)
            print("Standard Elapsed = ", time.time() - t0)
            t0 = time.time()
            covMatrix1, sumSpectrum1, nPixels1 = getCovarianceMatrix(stack,
                                                                   index=0,
                                                                   dtype='float64',
                                                                   force=True)
            print("Dynamic Elapsed = ", time.time() - t0)
            print(covMatrix0.max(), covMatrix0.min(), "Reference  = ", covMatrix0[10,50:60])
            print(covMatrix1.max(), covMatrix1.min(), "Calculated = ", covMatrix1[10,50:60])
            delta = covMatrix1-covMatrix0
            maxDiff = delta.max()
            print("Max diff   = ", maxDiff)
            minDiff = delta.min()
            print("Min diff   = ", minDiff)
            print(numpy.nonzero(delta==maxDiff))
            print("delta[0,79] = ", delta[0,79])
            print("reference[0,79] = ", covMatrix0[0,79])
            print("dynamic  [0,79] = ", covMatrix1[0,79])
            print("reference[79,0] = ", covMatrix0[79,0])
            print("dynamic  [79,0] = ", covMatrix1[79,0])
        else:
            #stack = EDFStack.EDFStack(inputfile, imagestack=True)
            from PyMca import DataObject
            import h5py
            f=h5py.File(inputfile, access='r')
            stack = DataObject.DataObject()
            stack.data = f['/data/NXdata/data']
            print("PCA Calculation")
            print stack.data.shape
            images, eigenvalues, eigenvectors = numpyPCA(stack, index=0, binning=1)
            for i in range(10):
                a = images[i]
                #a.shape = r, c
                print("Eigenvalue %d = %f" % (i, eigenvalues[i]))
                fname = "Image%02d.edf" % i
                if os.path.exists(fname):
                    os.remove(fname)
                edf = EdfFile.EdfFile(fname,'wb')
                edf.WriteImage({},a)
                edf = None
            inputfile = "D:\DATA\COTTE\CH1777\G4_mca_0012_0000_0000.edf"
            stack = EDFStack.EDFStack(inputfile,dtype=numpy.float64)
            images, eigenvalues, eigenvectors = numpyPCA(stack,
                                                         index=-1,
                                                         dtype='float64',
                                                         force=True,
                                                         binning=1)
            for i in range(10):
                a = images[i]
                #a.shape = r, c
                print("Eigenvalue %d = %f" % (i, eigenvalues[i]))
                fname = "Image%02d.edf" % (i+10)
                if os.path.exists(fname):
                    os.remove(fname)
                edf = EdfFile.EdfFile(fname,'wb')
                edf.WriteImage({},a)
                edf = None
    else:
        stack = EDFStack.EDFStack(inputfile, imagestack=False, dtype=numpy.float64)
        r, c, nChannels = stack.data.shape
        if 0:
            stack.data.shape = r * c, nChannels
            t0 = time.time()
            covMatrix0 = dotblas.dot(stack.data.T, stack.data)
            print("Standard Elapsed = ", time.time() - t0)
            print("Standard Shape = ", covMatrix0.shape)
            t0 = time.time()
            stack.data.shape = r , c, nChannels
            covMatrix1, sumSpectrum, nPixels = getCovarianceMatrix(stack,
                                                                   index=-1,
                                                                   dtype='float64',
                                                                   force=True)
            print("Dynamic Elapsed = ", time.time() - t0)
            print("Dynamic Shape = ", covMatrix1.shape)
            print(covMatrix0.max(), covMatrix0.min(), "Reference  = ", covMatrix0[1300, 1350:1360])
            print(covMatrix1.max(), covMatrix1.min(), "Calculated = ", covMatrix1[1300, 1350:1360])
            delta = covMatrix1-covMatrix0
            maxDiff = delta.max()
            print("Max diff   = ", maxDiff)
            minDiff = delta.min()
            print("Min diff   = ", minDiff)
            print(numpy.nonzero(delta==maxDiff))
            print("delta[79, 0] = ", delta[79, 0])
            print("reference[0,79] = ", covMatrix0[0,79])
            print("dynamic  [0,79] = ", covMatrix1[0,79])
            print("reference[79,0] = ", covMatrix0[79,0])
            print("dynamic  [79,0] = ", covMatrix1[79,0])
        else:
            print("PCA Calculation")
            images, eigenvalues, eigenvectors = numpyPCA(stack, index=-1)
            for i in range(10):
                a = images[i]
                #a.shape = r, c
                print("Eigenvalue %d = %f" % (i, eigenvalues[i]))
                fname = "Image%02d.edf" % i
                if os.path.exists(fname):
                    os.remove(fname)
                edf = EdfFile.EdfFile(fname,'wb')
                edf.WriteImage({},a)
                edf = None

    
