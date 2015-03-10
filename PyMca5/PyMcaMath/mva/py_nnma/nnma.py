#encoding:latin-1
#*/###########################################################################
# Copyright (c) 2009 Uwe Schmitt, uschmitt@mineway.de
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#    * notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
#    * copyright notice, this list of conditions and the following
#    * disclaimer in the documentation and/or other materials provided
#    * with the distribution.  Neither the name of the <ORGANIZATION>
#    * nor the names of its contributors may be used to endorse or
#    * promote products derived from this software without specific
#    * prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*/###########################################################################

import numpy as np
try:
    import scipy.sparse as sp
    has_sparse = True
    is_sparse = lambda A: isinstance(A, sp.spmatrix)

except ImportError:
    has_sparse = False
    is_sparse = lambda A: False

import math


__doc__ = """

py_nnma:  python modules for nonnegative matrix approximation (NNMA)

(c) 2009 Uwe Schmitt, uschmitt@mineway.de

NNMA minimizes  dist(Y, A X)

       where:  Y >= 0,  m x n
               A >= 0,  m x k
               X >= 0,  n x k

               k < min(m,n)

     dist(A,B) can be || A - B ||_fro
                   or   KL(A,B)


This moudule provides the following functions:

    NMF, NMFKL, SNMF, RRI, ALS, GDCLS, GDCLS_L1, FNMAI, FNMAI_SPARSE,
    NNSC and FastHALS

The common parameters when calling such a function are:

    input:

            Y           --   the matrix for decomposition, maybe dense
                             from numpy or sparse from scipy.sparse
                             package

            k           --   number of componnets to estimate

            Astart
            Xstart      --   matrices to start iterations. Maybe None
                             for using random start matrices.

            eps         --   termination swell value

            maxcount    --   max number of iterations to be performed

            verbose     --   if False: produce no output durint interations
                             if integer: give all 'verbose' itetations some
                             output about current state of iterations

    output:

            A, X        --   result matrices of algorithm

            obj         --   value of objective function of last iteration

            count       --   number of iterations done

            converged   --   flag: indicates if iterations stoped within
                             max number of iterations

The following extra parameters exist depending on algorithm:

    RRI      :  damping parameter 'psi' (default: 1e-12)

    SNMF     :  sparsity parameter 'sparse_par' (default: 0)

    ALS      :  regularization parameter 'regul' for stabilizing iterations
                (default value 0). needed if objective value jitters.

    GCDLS    :  'regul' for l2-smoothness of X (default 0)

    GDCLS_L1 :  'regul' for l1-smoothness of X (default 0)

    FNMAI    :  'stabil' for stabilizing algorithm (default value 1e-12)
                'alpha'  for stepsize  (default value 0.1)
                'tau'    for number of inner iterations (default value 2)

    FNMAI_SPARSE : as FNMAI plus
                'regul'  for l1-smoothness of X (default 0)

    NNSC     :  'alpha'       for stepsize of gradient update of A
                'sparse_par'  for sparsity

#############################################################################

Copyright (c) 2009 Uwe Schmitt, uschmitt@mineway.de

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
    * notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
    * copyright notice, this list of conditions and the following
    * disclaimer in the documentation and/or other materials provided
    * with the distribution.  Neither the name of the <ORGANIZATION>
    * nor the names of its contributors may be used to endorse or
    * promote products derived from this software without specific
    * prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


This module is based on:

    - Daniel D. Lee and H. Sebastian Seung:

          "Algorithms for non-negative matrix factorization",
          in Advances in Neural Information Processing 13
          (Proc. NIPS*2000) MIT Press, 2001.

          "Learning the parts of objects by non-negative matrix
           factorization",
          Nature, vol. 401, no. 6755, pp. 788-791, 1999.

    - A. Cichocki and A-H. Phan:

          "Fast local algorithms for large scale Nonnegative Matrix and
           Tensor Factorizations",
          IEICE Transaction on Fundamentals,
          in print March 2009.

    - P. O. Hoyer

          "Non-negative Matrix Factorization with sparseness
           constraints",
          Journal of Machine Learning Research, vol. 5, pp. 1457-1469,
          2004.


    - Dongmin Kim, Suvrit Sra,Inderjit S. Dhillon:

           "Fast Newton-type Methods for the Least Squares Nonnegative Matrix
           Approximation Problem"
           SIAM Data Mining (SDM), Apr. 2007


    - Ngoc-Diep Ho:

        dissertation from
        http://edoc.bib.ucl.ac.be:81/ETD-db/collection/available/BelnUcetd-06052008-235205/

"""
__license__ = "BSD"
#
# helper functions for handling sparse and dense matrices from numpy
# and scipy.sparse
#
def divide_sparse_matrix(A, by):

    assert isinstance(A, sp.spmatrix), "wrong format"

    A = A.tocoo()
    A.data /= by[A.row, A.col]
    return A

def divide_matrix(A, by):

    if is_sparse(A):
        return divide_sparse_matrix(A, by)
    elif isinstance(A, np.ndarray):
        return A / by
    else:
        raise TypeError("wrong matrix format %s" % type(A))

def dot(A, B):

    if is_sparse(A) and is_sparse(B):
        return (A*B).todense()
    elif is_sparse(A):
        return A*B
    elif is_sparse(B):
        return (B.transpose() * A.T).T
    else:
        return np.dot(A, B)

def diff(A, B):
    E = A - B
    # if A is sparse E is np.matrix
    # if A is dense  E is np.ndarray
    # so: convert np.matrix to np.ndarray if needed:
    if isinstance(E, np.matrix):
        return E.A
    return E

def flatten(A):

    if is_sparse(A):
        return A.todense().flatten().A
    else:
        return A.flatten()

def frob_norm(A):
    if is_sparse(A):
        return math.sqrt( (A.data**2).sum())
    else:
        return np.linalg.norm(A)

def transpose(A):

    if is_sparse(A):
        return  sp.csr_matrix(A.transpose())
    else:
        return A.T

def get_scaling_vector(A, p=1.0):

    if is_sparse(A):
        dd = ((A**p).tocsc().sum(axis=0).A)**(1.0/p)
    else:
        dd = ((A**p).sum(axis=0))**(1.0/p)
    return dd

def coerced(Y):

    # csr is faster for matrix-vector or matrix-matrix products
    if is_sparse(Y):

        if isinstance(Y, sp.csc_matrix):
            YT = sp.csr_matrix(Y.T)
            Y  = sp.csr_matrix(Y)

        elif isinstance(Y, sp.csr_matrix):
            YT = sp.csr_matrix(Y.T)

    elif isinstance(Y, np.ndarray):
        YT = Y.T

    return Y, YT

#
#    building blocks for nnma algorithms
##


def GradA(Y, YT, A, X, **param):
    """ dPhi(Y, A, X) / dA  with  Phi(Y, A, X) = || Y - A X ||_fro """

    XXT = np.dot(X, X.T)
    return np.dot(A, XXT) - dot(Y, X.T)


def GradX(Y, YT ,A, X, **param):
    """ dPhi(Y, A, X) / dX  with  Phi(Y, A, X) = || Y - A X ||_fro """

    ATA = np.dot(A.T, A)
    return np.dot(ATA, X) - dot(A.T, Y)

def GradA_step(Y, YT, A, X, **param):

    alpha = param.get("alpha", 1e-3)
    A = A - alpha * GradA(Y, YT, A, X, **param)
    #A /= np.sqrt((A*A).sum(axis=0))
    A[A<0] = 0
    return A

def GradX_step(Y, YT, A, X, **param):

    alpha = param.get("alpha", 1e-6)
    X = X - alpha * GradX(Y, YT, A, X, **param)
    X[X<0] = 0
    return X

def A_mult_update_kl_div(Y, YT, A, X, **param):
    """ update A for minimization of KL(Y || A X) """

    AX = np.dot(A, X)
    Y_by_AX = divide_matrix(Y, 1e-9+AX)
    F = dot(Y_by_AX, X.T) / X.sum(axis=1).T
    return A*F

def X_mult_update_kl_div(Y, YT, A, X, **param):
    """ update V for minimization of KL(Y || A X) """

    AX = np.dot(A, X)
    Y_by_AX = divide_matrix(Y, 1e-9+AX)

    F = dot(transpose(Y_by_AX), A).T
    return X* (F.T / A.sum(axis=0)).T

def A_mult_update(Y, YT, A, X, **param):
    """ Lee and Sung multiplicative update """

    AXXT = np.dot(A, np.dot(X, X.T))
    F = dot(Y, X.T)/(1e-9 + AXXT)
    return A*F

def X_mult_update(Y, YT, A, X, **param):
    """ Lee and Sung multiplicative update """

    ATAX = np.dot(np.dot(A.T, A),X)
    ATY  = dot(YT, A).T
    F = ATY/(1e-9 + ATAX)
    return X*F

def X_mult_update_nnsc(Y, YT, A, X, **param):
    """ Lee and Sung multiplicative update """

    regul=param.get("sparse_par", 1e-9)
    ATAX = np.dot(np.dot(A.T, A),X)
    ATY  = dot(YT, A).T
    F = ATY/(regul + ATAX)
    return X*F

def A_inexact_lsq_update(Y, YT, A, X, **param):
    """ ALS fixed point update """

    regul=param.get("regul", 0.0)

    XXT = np.dot(X, X.T)
    YXT = dot(Y, X.T)
    A =  np.dot(YXT,  np.linalg.pinv(XXT + regul*np.eye(XXT.shape[0])))
    A[A<0] = 0
    return A

def X_inexact_lsq_update(Y, YT, A, X, **param):
    """ ALS fixed point update """

    regul=param.get("regul", 0.0)

    ATA = np.dot(A.T, A)
    ATY  = dot(YT, A).T
    X = np.dot(np.linalg.pinv(ATA + regul*np.eye(ATA.shape[0])), ATY)
    X[X<0] = 0
    return X

def X_inexact_lsq_update_l1regul(Y, YT, A, X, **param):
    """ ALS fixed point update with L1 regularization for X """

    regul=param.get("regul", 0.0)
    ATA = np.dot(A.T, A)
    ATY  = dot(YT, A).T
    X = np.dot(np.linalg.pinv(ATA+1e-12*np.eye(ATA.shape[0])),ATY-regul)
    X[X<0] = 0
    return X

def FNMAI_A_update(Y, YT, A, X, **param):
    """ FNMAI (Kim et al) update for A """

    stabil=param.get("stabil", 1e-12)
    alpha=param.get("alpha", 0.1)
    tau=param.get("tau", 2)
    k = A.shape[1]
    a = max(1e-9, stabil)

    for _ in range(tau):
        G = GradA(Y, YT, A, X)
        Iplus = (A==0) & (G>0)
        G[Iplus] = 0

        G = np.dot(G, np.linalg.pinv(np.dot(X,X.T)+a*np.eye(k)))
        G[Iplus] = 0
        A -= alpha*G
        A[A<0] = 0
    return A

def FNMAI_X_update(Y, YT, A, X, **param):
    """ FNMAI (Kim et al) update for V """

    stabil=param.get("stabil", 1e-12)
    alpha=param.get("alpha", 0.1)
    tau=param.get("tau", 2)
    k = A.shape[1]
    a = max(1e-9, stabil)

    for _ in range(tau):
        G = GradX(Y, YT, A, X)
        Iplus = (X==0) & (G>0)
        G[Iplus] = 0

        G = np.dot(np.linalg.pinv(np.dot(A.T,A)+a*np.eye(k)), G)
        G[Iplus] = 0
        X -= alpha*G
        X[X<0] = 0
    return X

def FastHALS_X_update(Y, YT, A, X, **param):
    W = dot(YT, A)
    V = dot(A.T, A)
    k = A.shape[1]
    for i in range(k):
        xi = X[i,:]
        xi += W[:,i]-dot(X.T, V[:,i])
        xi[xi<0] = 0
        X[i,:] = xi
    return X

def FastHALS_A_update(Y, YT, A, X, **param):
    P = dot(Y, X.T)
    Q = dot(X, X.T)
    k = A.shape[1]
    for i in range(k):
        ai = A[:,i]
        ai = ai*Q[i,i] + P[:,i]-dot(A, Q[:,i])
        ai[ai<0] = 0
        ai /= np.linalg.norm(ai)
        A[:,i] = ai

    return A

#
# All NNMA algorithms have the same structure which is implemented
# in AlgorunnerTemplate
#

class AlgorunnerTemplate(object):

    def frob_dist(self, Y, A, X):
        """ frobenius distance between Y and A X """
        return np.linalg.norm(Y - np.dot(A,X))

    def kl_divergence(self, Y, A, X):
        """ kullbach leibler divergence D(Y | A X) """
        AXvec = np.dot(A, X).flatten()
        Yvec = flatten(Y)

        return (Yvec*np.log(Yvec/AXvec)-Yvec+AXvec).sum()

    dist = frob_dist # default case

    def init_factors(self, Y, k,  A=None, X=None):
        """ generate start matrices U, V """

        m, n = Y.shape

        # sample start matrices
        if A is None:
            A = np.random.rand(m,k)
        elif isinstance(A, np.matrix):
            A = A.A
        if X is None:
            X = np.random.rand(k,n)
        elif isinstance(X, np.matrix):
            X = X.A

        # scale A, X with alpha such that || Y - alpha AX ||_fro is
        # minimized

        AX = np.dot(A,X).flatten()
        # alpha = < Y.flatten(), AX.flatten() > / < AX.flatten(),AX.flatten() >
        if is_sparse(Y):
            # can we improve this confirming memory usage ????
            alpha = np.diag(dot(Y, np.dot(A,X).T)).sum()/np.dot(AX, AX)
        else:
            alpha = np.dot(Y.flatten(), AX)/np.dot(AX,AX)

        A /= math.sqrt(alpha)
        X /= math.sqrt(alpha)

        return A, X

    param_update = None  # default, may be overidden by method which
                         # adapts parametes from iteration to iteration

    def __call__(self, Y, k, A=None, X=None, eps=1e-5,
                 maxcount=1000, verbose=False, **param):

        """ basic template for NNMA iterations """

        m, n = Y.shape

        if k<1 or k>m or k>n:
            raise ValueError("number k of components is invalid")

        Y, YT = coerced(Y)

        A, X = self.init_factors(Y, k, A, X)

        count = 0
        obj_old = 1e99

        param = param.copy()

        # works for sparse and for dense matrices:
        # calculate frobenius norm of Y
        nrm_Y = frob_norm(Y)

        while True:

            A, X = self.update(Y, YT, A, X, **param)

            if np.any(np.isnan(A)) or np.any(np.isinf(A)) or \
               np.any(np.isnan(X)) or np.any(np.isinf(X)):

                if verbose:
                    print("RESTART")
                A, X = self.init_factors(Y, k)
                count = 0

            count += 1

            # relative distance which is independeant to scaling of A
            obj = self.dist(Y, A, X) / nrm_Y

            delta_obj = obj-obj_old
            if verbose:
                # each 'verbose' iterations report about actual state
                if count % verbose == 0:
                    print("count=%6d obj=%E d_obj=%E" %(count, obj,
                                                        delta_obj))

            if count >= maxcount: break
            # delta_obj should be "almost negative" and small enough:
            if -eps < delta_obj <= 1e-12:
                break

            obj_old = obj
            if self.param_update is not None:
                self.param_update(param)

        if verbose:
            print("FINISHED:")
            print("count=%6d obj=%E d_obj=%E" %(count, obj, delta_obj))

        return A, X,  obj, count, count < maxcount


#
# Most NNMA algorithms have global updates of U and V which can be
# combined with the following base class:
#

class FactorizedNNMA(AlgorunnerTemplate):

    def __init__(self, update_A, update_X, param_update = None):
        self.update_A = update_A
        self.update_X = update_X
        self.param_update = param_update

    def update(self, Y,  YT, A, X,  **param):

        A = self.update_A(Y, YT, A, X, **param)
        X = self.update_X(Y, YT, A, X, **param)

        return A, X

class SNMF_(AlgorunnerTemplate):

    """
    W. Liu, N. Zheng, and X. Lu.:
    "Non-negative matrix factorization for visual coding". In Proc. IEEE Int.
    Conf. on Acoustics, Speech and Signal Processing (ICASSP’2003), 2003
    """

    # use kullbach-level distance
    dist = AlgorunnerTemplate.kl_divergence

    def update(self, Y, YT, A, X, **param):

        sparse_par = param.get("sparse_par", 0.0)

        A /= A.sum(axis=0)+1e-9
        AX = np.dot(A, X)

        Y_by_AX = divide_matrix(Y, 1e-9+AX)

        X *= dot(Y_by_AX.T, A).T / (1.0 + sparse_par)

        AX = np.dot(A, X)
        Y_by_AX = divide_matrix(Y, 1e-9+AX)
        F = dot(Y_by_AX, X.T) / ( X.T.sum(axis=0) + 1e-9)
        A *= F

        return A, X

class RRI_(AlgorunnerTemplate):

    """
    Runtime optimisations from Cichocki applied to
    Damped rank one residual iteration from Ngoc-Diep Ho.
    """

    def update(self, Y, YT, A, X,  **param):

        E = diff(Y, np.dot(A,X))

        psi = param.get("psi", 1e-12)

        for j in range(A.shape[1]):

            aj = A[:,j]
            xj = X[j,:]

            Rt =  E + np.outer(aj, xj)

            xj = np.dot(Rt.T, aj)+psi*xj
            xj[xj<0]= 0

            fac = np.linalg.norm(aj)**2
            xj /= fac+psi

            aj = np.dot(Rt, xj)+psi*aj
            aj[aj<0]= 0

            fac = np.linalg.norm(xj)**2
            aj /= fac+psi

            A[:,j] = aj
            X[j,:] = xj

            E = Rt - np.outer(aj, xj)

        return A, X


#
# create  algorithms objects
#

SNMF     = SNMF_()
RRI      = RRI_()

# classical algorithme with frobenius norm for calculating
# objective function
NMF      = FactorizedNNMA(A_mult_update, X_mult_update)

# classical algorithme with kl divergence or calculating
# objective function
NMFKL    = FactorizedNNMA(A_mult_update_kl_div, X_mult_update_kl_div)


# Stabilized alternating least sqaures with decreasing regularization
# from Cichocki et al.

def regul_dec(param):
    param["regul"] = param.get("regul", 0)* .9

ALS      = FactorizedNNMA(A_inexact_lsq_update, X_inexact_lsq_update,
                          regul_dec)
# GDCLS from
# "Document clustering using nonnegative matrix factorization"
# Information Processing and Management
# Volume 42 ,  Issue 2  (March 2006) t
# Pages: 373 - 386  ,
GDCLS    = FactorizedNNMA(A_mult_update, X_inexact_lsq_update)

#Fast Newton-type Method from Kim et al
FNMAI    = FactorizedNNMA(FNMAI_A_update, FNMAI_X_update)


# own algorithms for approximation of Y ~ A X

# replace l2-regularisation when updating X by l1-regularization
# for getting spare coordinates
GDCLS_L1 = FactorizedNNMA(A_mult_update, X_inexact_lsq_update_l1regul)

# replace FNMAI_X_update by l1 regulraized least squares update
FNMAI_SPARSE = FactorizedNNMA(FNMAI_A_update, \
                              X_inexact_lsq_update_l1regul)

# Hoyers sparse coding algorithm
NNSC = FactorizedNNMA(GradA_step, X_mult_update_nnsc)

# FastHALS from Cichocki and Phan
FastHALS = FactorizedNNMA(FastHALS_A_update, FastHALS_X_update)

if __name__ == "__main__":

    # test all routines !

    param = dict(alpha=.1, tau=2, regul=1e-2, sparse_par=1e-1, psi=1e-3)

    nc = 10
    B = np.random.rand(30,nc)
    C = np.random.rand(nc,20)
    A = np.dot(B, C)

    import sys, time

    def run(name, routine, verbose=0):
        print("run %12s" % name,)
        sys.stdout.flush()
        start = time.time()
        X,Y,obj,count,converged = routine(A, 10, eps=5e-5, verbose=verbose,
                                          maxcount=1000, **param)
        print("obj = %E  count=%5d  converged=%d  TIME=%.2f secs" % \
                     (obj,count, converged, time.time()-start))

    print("\nTEST WITH DENSE MATRIX\n")

    run("NNSC", NNSC, verbose=0)
    run("FNMAI_SPARSE", FNMAI_SPARSE)
    run("FNMAI", FNMAI)
    run("GDCLS_L1", GDCLS_L1)
    run("GDCLS", GDCLS)
    run("ALS", ALS)
    run("NMFKL", NMFKL)
    run("NMF", NMF)
    run("RRI", RRI)
    run("FastHALS", FastHALS)
    run("SNMF", SNMF)


    if has_sparse:
        print("\nTEST WITH SPARSE MATRIX\n")
        A = sp.csc_matrix(A)

        run("NNSC", NNSC, verbose=0)
        run("FNMAI_SPARSE", FNMAI_SPARSE)
        run("FNMAI", FNMAI)
        run("GDCLS_L1", GDCLS_L1)
        run("GDCLS", GDCLS)
        run("ALS", ALS)
        run("NMFKL", NMFKL)
        run("NMF", NMF)
        run("RRI", RRI)
        run("FastHALS", FastHALS)
        run("SNMF", SNMF)
