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
__author__ = "V.A. Sole - ESRF BLISS Group, A. Mirone - ESRF SciSoft Group"
import numpy
import numpy.linalg
try:
    import numpy.core._dotblas as dotblas
except ImportError:
    dotblas = numpy
    
try:
    import mdp
    MDP = True
except ImportError:
    MDP = False
import Lanczos
import os
DEBUG = 0

def lanczosPCA(stack, ncomponents, binning=None):
    if DEBUG:
        print "lanczosPCA"
    if binning is None:
        binning = 1
        
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack

    #wrappmatrix = "double" 
    wrapmatrix = "single" 

    dtype = numpy.float64
    if wrapmatrix == "double":
        data = data.astype(dtype)

    if len(data.shape) == 3:
        r, c, N = data.shape
        data.shape = r*c, N
    else:
        r, N = data.shape
        c = 1
        
    npixels = r * c
    
    if binning > 1:
        data=numpy.reshape(data,[data.shape[0], data.shape[1]/binning, binning])
        data=numpy.sum(data , axis=-1)
        N=N/binning

    if ncomponents > N:
        raise ValueError, "Number of components too high."

    avg = numpy.sum(data, 0)/(1.0*npixels)
    numpy.subtract(data, avg, data)

    Lanczos.LanczosNumericMatrix.tipo=dtype
    Lanczos.LanczosNumericVector.tipo=dtype


    if wrapmatrix=="single" :
        SM=[dotblas.dot(data.T, data).astype(dtype)]
        SM = Lanczos.LanczosNumericMatrix( SM )
    else:
        SM = Lanczos.LanczosNumericMatrix( [data.T.astype(dtype), data.astype(dtype) ])
    
    eigenvalues, eigenvectors = Lanczos.solveEigenSystem( SM,
                                                          ncomponents,
                                                          shift=0.0,
                                                          tol=1.0e-15)
    SM = None
    numpy.add(data, avg, data)

    images = numpy.zeros((ncomponents, npixels), data.dtype)
    vectors = numpy.zeros((ncomponents, N), dtype)
    for i in range(ncomponents):
        vectors[i, :] = eigenvectors[i].vr
        images[i,:] = dotblas.dot(data, (eigenvectors[i].vr).astype(data.dtype))
    data = None
    images.shape = ncomponents, r, c
    return images, eigenvalues, vectors

def lanczosPCA2(stack, ncomponents, binning=None):
    """
    This is a fast method, but it may loose information
    """
    binning = 1
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    r, c, N = data.shape

    #data=Numeric.fromstring(data.tostring(),"f")


    npixels = r * c 		#number of pixels
    data.shape = r*c, N

    if npixels < 2000:
        BINNING = 2
    if npixels < 5000:
        BINNING = 4
    elif npixels < 10000:
        BINNING = 8
    elif npixels < 20000:
        BINNING = 10
    elif npixels < 30000:
        BINNING = 15
    elif npixels < 60000:
        BINNING = 20
    else:
        BINNING = 30
    if BINNING is not None:
        dataorig = data
        reminder = npixels % BINNING
        if reminder:
            data = data[0:BINNING*int(npixels/BINNING),:]
        data=numpy.reshape(data,[data.shape[0]/BINNING,  BINNING,  data.shape[1] ])
        data = numpy.swapaxes(data,1,2)
        data=numpy.sum(data , axis=-1)
        rc=int(r*c/BINNING)

    ##########################################
    tipo=numpy.float64
    neig=ncomponents + 5
    rappmatrix = "doppia" #non crea la matrice de covarianza ma fa due multiplica.
    rappmatrix = "singola" #crea la matrice de covarianza ma fa soltanto una multiplica.
    ######################################


    # calcola la media
    mediadata = numpy.sum(data, axis = 0) / numpy.array([len(data)],data.dtype)

    numpy.subtract(data,mediadata,data)

    Lanczos.LanczosNumericMatrix.tipo=tipo
    Lanczos.LanczosNumericVector.tipo=tipo



    if rappmatrix=="singola" :
        SM=[dotblas.dot(data.T, data).astype(tipo)]
        SM = Lanczos.LanczosNumericMatrix( SM )
    else:
        SM =Lanczos.LanczosNumericMatrix( [data.T.astype(tipo), data.astype(tipo) ])

    ev,eve=Lanczos.solveEigenSystem( SM, neig, shift=0.0, tol=1.0e-7)
    SM = None
    rc = rc*BINNING

    newmat = numpy.zeros([ r*c, neig ], numpy.float64   )
    dumadd = numpy.zeros([ r*c, neig ], numpy.float64   )

    #datadiff = Numeric.array(data)
    #print " CHI = " , Numeric.sum(Numeric.sum(datadiff*datadiff))

    data=data.astype(tipo)

    numpy.add(data,mediadata,data)

    # print " add " 
    # ??????????????????//  Numeric.add(data, mediadata, data)
    # print " add OK "

    for i in range(neig):
        newmat[:,i] = dotblas.dot(dataorig, (eve[i].vr).astype(dataorig.dtype))

    newcov =   dotblas.dot(  newmat.T, newmat )
    evals, evects = Lanczos.LinearAlgebra.Heigenvectors(newcov)

    nuovispettri = dotblas.dot(  evects , eve.vr[:neig] )
    images = numpy.zeros((ncomponents, npixels), data.dtype)
    vectors = numpy.zeros((ncomponents, N), tipo)
    for i in range(ncomponents):
        vectors[i,:] = nuovispettri[-1-i,:]
        images[i,:] = dotblas.dot(  newmat , evects[-1-i].astype(dataorig.dtype)  )
    images.shape = ncomponents, r, c
    return images, evals,  vectors

def expectationMaximizationPCA(stack, ncomponents, binning=None):
    """
    This is a fast method when the number of components is small
    """
    if DEBUG:
        print "expectationMaximizationPCA"
    #This part is common to all ...
    if binning is None:
        binning = 1
        
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    if len(data.shape) == 3:
        r, c, N = data.shape
        data.shape = r*c, N
    else:
        r, N = data.shape
        c = 1

    if binning > 1:
        data=numpy.reshape(data,[data.shape[0], data.shape[1]/binning, binning])
        data=numpy.sum(data , axis=-1)
        N=N/binning
    if ncomponents > N:
        raise ValueError, "Number of components too high."
    #end of common part
    avg = numpy.sum(data, 0)/(1.0*r*c)
    numpy.subtract(data, avg, data)
    dataw = data * 1
    images = numpy.zeros((ncomponents, r * c), data.dtype)
    eigenvalues = numpy.zeros((ncomponents,), data.dtype)
    eigenvectors = numpy.zeros((ncomponents, N), data.dtype)
    for i in range(ncomponents):
        #generate a random vector
        p = numpy.random.random(N)
        #10 iterations seems to be fairly accurate, but it is
        #slow when reaching "noise" components.
        #A variation threshold of 1 % seems to be acceptable.
        tmod_old = 0
        tmod = 0.02
        j = 0
        max_iter = 7
        while ((abs(tmod-tmod_old)/tmod) > 0.01) and (j<max_iter):
            tmod_old = tmod
            t = 0.0
            for k in range(r*c):
                t += dotblas.dot(dataw[k,:],p.T) * dataw[k,:]
            tmod = numpy.sqrt(numpy.sum(t*t))
            p = t/tmod
            j+=1
        #print "Iterations = ", j, 'last per cent variation', 100*(abs(tmod-tmod_old)/tmod)
        eigenvectors[i, :] = p
        #subtract the found component from the dataset
        for k in range(r*c):
            dataw[k,:] -= dotblas.dot(dataw[k,:],p.T) * p
        #print "One component calculated"
    # calculate eigenvalues via the Rayleigh Quotients:
    # eigenvalue = (Eigenvector.T * Covariance * EigenVector)/ (Eigenvector.T * Eigenvector)
    for i in range(ncomponents):
        tmp = dotblas.dot(data, eigenvectors[i,:].T)
        eigenvalues[i] = dotblas.dot(tmp.T, tmp)/dotblas.dot(eigenvectors[i,:].T, eigenvectors[i,:])

    #Generate the eigenimages
    for i0 in range(ncomponents):
        images[i0,:] = dotblas.dot(data , eigenvectors[i0,:])

    #restore the original data
    numpy.add(data, avg, data)

    #reshape the images
    images.shape = ncomponents, r, c
    return images, eigenvalues, eigenvectors

def numpyPCA(stack, ncomponents, binning=None):
    """
    This is a covariance method using numpy numpy.linalg.eigh
    """
    #This part is common to all ...
    if binning is None:
        binning = 1
        
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    if len(data.shape) == 3:
        r, c, N = data.shape
        data.shape = r*c, N
    else:
        r, N = data.shape
        c = 1

    if binning > 1:
        data=numpy.reshape(data,[data.shape[0], data.shape[1]/binning, binning])
        data=numpy.sum(data , axis=-1)
        N=N/binning
    if ncomponents > N:
        raise ValueError, "Number of components too high."
    #end of common part

    #begin the specific coding
    avg = numpy.sum(data, 0)/(1.0*r*c)
    numpy.subtract(data, avg, data)
    cov = numpy.dot(data.T, data)
    evalues, evectors = numpy.linalg.eigh(cov)
    cov = None
    images = numpy.zeros((ncomponents, r * c), data.dtype)
    eigenvalues = numpy.zeros((ncomponents,), data.dtype)
    eigenvectors = numpy.zeros((ncomponents, N), data.dtype)
    #sort eigenvalues
    a = [(evalues[i], i) for i in range(len(evalues))]
    a.sort()
    a.reverse()
    for i0 in range(ncomponents):
        i = a[i0][1]
        eigenvalues[i0] = evalues[i]
        eigenvectors[i0,:] = evectors[:,i]
        images[i0,:] = dotblas.dot(data , eigenvectors[i0,:])
    
    #restore the original data
    numpy.add(data, avg, data)

    #reshape the images
    images.shape = ncomponents, r, c
    return images, eigenvalues, eigenvectors

def mdpPCASVDFloat32(stack, ncomponents, binning=None):
    return mdpPCA(stack, ncomponents,
                  binning=binning, dtype='float32', svd='True')

def mdpPCASVDFloat64(stack, ncomponents, binning=None):
    return mdpPCA(stack, ncomponents,
                  binning=binning, dtype='float64', svd='True')

def mdpPCA(stack, ncomponents, binning=None, dtype='float64', svd='True'):
    if DEBUG:
        print "MDP Method"
        print "binning =", binning
        print "dtype = ", dtype
        print "svd = ", svd
    #This part is common to all ...
    if binning is None:
        binning = 1

    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    if len(data.shape) == 3:
        r, c, N = data.shape
        data.shape = r*c, N
    else:
        r, N = data.shape
        c = 1

    if binning > 1:
        data=numpy.reshape(data,[data.shape[0], data.shape[1]/binning, binning])
        data=numpy.sum(data , axis=-1)
        N=N/binning
    if ncomponents > N:
        raise ValueError, "Number of components too high."
    #end of common part

    #begin the specific coding
    pca = mdp.nodes.PCANode(output_dim=ncomponents, dtype=dtype, svd=svd)
    pca.train(data)

    pca.stop_training()

    avg = pca.avg
    eigenvalues = pca.d
    eigenvectors = pca.v.T
    proj = pca.get_projmatrix()
    images = numpy.dot((proj.T).astype(data.dtype), data.T)

    #reshape the images
    images.shape = ncomponents, r, c
    return images, eigenvalues, eigenvectors

def mdpICA(stack, ncomponents, binning=None):
    if binning is None:
        binning = 1

    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    if len(data.shape) == 3:
        r, c, N = data.shape
        data.shape = r*c, N
    else:
        r, N = data.shape
        c = 1

    if binning > 1:
        data=numpy.reshape(data,[data.shape[0], data.shape[1]/binning, binning])
        data=numpy.sum(data , axis=-1)
        N=N/binning
    if ncomponents > N:
        raise ValueError, "Number of components too high."
    if 0:
        pca = mdp.nodes.PCANode(output_dim=ncomponents, dtype='float64')
        pca.train(data)

        pca.stop_training()

        avg = pca.avg
        eigenvalues = pca.d
        eigenvectors = pca.v.T
        proj = pca.get_projmatrix(transposed=0)
        images = numpy.dot(proj.astype(data.dtype), data.T)    
        images.shape = ncomponents, r, c
    else:
        ica = mdp.nodes.FastICANode(white_comp=ncomponents, verbose=False, dtype='float64')
        ica.train(data)
        output = ica.execute(data)

        proj = ica.get_projmatrix(transposed=0)
        #print dir(ica)
        #print ica.filters.shape
        #print ica.mu
        #print ica.get_output_dim()
        #print dir(ica.white)
        #print output[0]
        
        icacomponents = proj


        # These are the PCA data
        eigenvalues = ica.white.d
        eigenvectors = ica.white.v.T
        images = numpy.zeros((2*ncomponents, r * c), data.dtype)
        vectors = numpy.zeros((ncomponents*2, N), data.dtype)
        vectors[0:ncomponents,:] = proj #ica components?
        vectors[ncomponents:,:] = eigenvectors
        images[0:ncomponents,:] = numpy.dot(proj.astype(data.dtype), data.T)    
        proj = ica.white.get_projmatrix(transposed=0)
        images[ncomponents:(2*ncomponents),:] = numpy.dot(proj.astype(data.dtype), data.T)    
        images.shape = 2 * ncomponents, r, c
        
    return images, eigenvalues, vectors


if __name__ == "__main__":
    import EDFStack
    import EdfFile
    import os
    import sys
    import time
    #inputfile = ".\PierreSue\CH1777\G4-Sb\G4_mca_0012_0000_0000.edf"
    inputfile = ".\COTTE\ch09\ch09__mca_0005_0000_0000.edf"    
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
        print inputfile
    elif os.path.exists(inputfile):
        print "Using a default test case"
    else:
        print "Usage:"
        print "python PCAModule.py indexed_edf_stack"
        sys.exit(0)
    stack = EDFStack.EDFStack(inputfile)
    r0, c0, n0 = stack.data.shape
    ncomponents = 10
    outfile = os.path.basename(inputfile)+"PCA.edf"
    e0 = time.time()
    images, eigenvalues, eigenvectors =  lanczosPCA2(stack.data, ncomponents,
                                                     binning=1)
    stack.data.shape = r0, c0, n0
    print "PCA Elapsed = ", time.time() - e0
    #print "eigenvalues PCA1 = ", eigenvalues
    print "eigenvectors PCA2 = ", eigenvectors[0,200:230]
    #stack = EDFStack.EDFStack(inputfile)
    #stack = EDFStack.EDFStack(inputfile)
    stack = None
    stack = EDFStack.EDFStack(inputfile)
    e0 = time.time()
    images2, eigenvalues, eigenvectors =  mdpPCA(stack.data, ncomponents,
                                                     binning=1)
    stack.data.shape = r0, c0, n0
    print "MDP Elapsed = ", time.time() - e0
    #print "eigenvalues MDP = ", eigenvalues
    print "eigenvectors MDP = ", eigenvectors[0,200:230]
    if os.path.exists(outfile):
        os.remove(outfile)
    f = EdfFile.EdfFile(outfile)
    for i in range(ncomponents):
        f.WriteImage({}, images[i,:])
    for i in range(ncomponents):
        f.WriteImage({}, images2[i,:])
    f = None
