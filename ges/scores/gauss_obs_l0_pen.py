# Copyright 2020 Juan Luis Gamella Martin

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
"""

import numpy as np
from .decomposable_score import DecomposableScore

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment
class GaussObsL0Pen(DecomposableScore):
    
    def __init__(self, data, lmbda=None, method='scatter', cache=True, debug=0):
        if type(data) != np.ndarray:
            raise ValueError("Provided data has the wrong format, type(data)=%s" % type(data))

        super().__init__(data, cache=cache, debug=debug)
        
        self.n, self.p = data.shape
        self.lmbda = 0.5 * np.log(self.n) if lmbda is None else lmbda
        self.method = method
        
        # Precompute scatter matrices if necessary
        if method == 'scatter':
            self._scatter = np.cov(data, rowvar=False, ddof=0)
        elif method == 'raw':
            self.data = np.hstack([data, np.ones((self.n, 1))])
        else:
            raise ValueError('Unrecognized method "%s"' % method)

    def full_score(self, A):
        """
        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a sample from a single environment, by finding the maximum
        likelihood estimates of the corresponding connectivity matrix
        (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j

        Returns
        -------
        score : float
            the penalized log-likelihood score

        """
        # Compute MLE
        B,omegas = self._mle_full(A)
        # Compute log-likelihood (without log(2π) term)
        K = np.diag(1/omegas)
        det_K = np.prod(1/omegas)
        I_B = np.eye(self.p) - B.T
        likelihood = 0.5 * self.n * (np.log(det_K) - np.trace(K @ I_B @ self._scatter @ I_B.T))
        #   Note: the number of parameters is the number of edges + the p marginal variances
        l0_term = self.lmbda * (np.sum(A != 0) + 1*self.p)
        score = likelihood - l0_term
        return score    

    # Note: self.local_score(...), with cache logic, already defined
    # in parent class DecomposableScore.
    
    def _compute_local_score(self, x, pa):
        """
        Given a node and its parents, return the local l0-penalized
        log-likelihood of a sample from a single environment, by finding
        the maximum likelihood estimates of the weights and noise term
        variances.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the penalized log-likelihood score

        """
        pa = list(pa)
        # Compute MLE
        b, sigma = self._mle_local(x, pa)
        # Compute log-likelihood (without log(2π) term)
        likelihood = -0.5*self.n*( 1 + np.log(sigma))
        #  Note: the number of parameters is the number of parents (one
        #  weight for each) + the marginal variance of x
        l0_term = self.lmbda * (len(pa) + 1)
        score = likelihood - l0_term
        return score
    
    # --------------------------------------------------------------------
    #  Functions for the maximum likelihood estimation of the
    #  weights/variances

    def _mle_full(self, A):
        B = np.zeros(A.shape)
        omegas = np.zeros(self.p)
        for j in range(self.p):
            parents = np.where(A[:,j] != 0)[0]
            B[:,j], omegas[j] = self._mle_local(j, parents)
        return B, omegas

    def _mle_local(self, j, parents):
        parents = list(parents)
        b = np.zeros(self.p)
        # Compute the regression coefficients from a least squares
        # regression on the raw data
        if self.method == 'raw':
            X = np.atleast_2d(self.data[:, parents + [self.p]]) # [p] for the intercept
            Y = data[:,j]
            # Perform regression
            coef = np.linalg.lstsq(X,Y, rcond=None)
            b[parents] = coef[:-1]
            #intercept = coef[-1]
            sigma = np.var(y - X @ coef)
        # Or compute the regression coefficients from the
        # empirical covariance (scatter) matrix
        # i.e. b = Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
        elif self.method == 'scatter':
            sigma = self._scatter[j,j]
            if len(parents) > 0:
                cov_parents = self._scatter[parents, :][:, parents]
                cov_j = self._scatter[j, parents]
                # Use solve instead of inverting the matrix
                coef = np.linalg.solve(cov_parents, cov_j)
                sigma = sigma - cov_j @ coef
                b[parents] = coef
        return b, sigma
