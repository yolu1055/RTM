import sys, re, time, string
import numpy as np
from scipy.special import gammaln, psi
import readfiles
from scipy import linalg as LA
from rexec import RHooks

np.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

class RTM():
    def __init__(self, alpha, K, A, rho, D, V, ids, cts, max_it):
        self._K = K
        self._alpha = alpha
        self._rho = rho
        self._A =A
        self._D = D
        self._V = V
        self._M = 0.0
        for name in A.keys():
            self._M = self._M + len(self._A[name])
            
        self._M = self._M / 2.0
        
        self._max_it = max_it
        
        self._nu = 0
        self._eta = np.random.normal(0., 1, self._K)
        
        self._beta = 1*np.random.gamma(100., 1./100., (self._K, self._V))
        self._beta = self._beta / np.sum(self._beta, axis = 1)[:, np.newaxis]
        
        
        self._N = {}
        self._phi = {}
        self._names = ids.keys()
        self._pi = {}
        self._gamma = {}
        
        for name in self._names:
            nd = np.sum(cts[name])
            self._N[name] = nd
            self._phi[name] = 1*np.random.gamma(100., 1./100., (len(ids[name]), self._K))
            self._pi[name] = np.sum(cts[name] * self._phi[name].T, axis = 1) / nd
            self._gamma[name] = 1*np.random.gamma(100., 1./100., self._K)
            
            
    def do_e_step(self, ids, cts):
        sstats = np.zeros((self._K, self._V))
        for name in self._names:
            gammad = 1*np.random.gamma(100., 1./100., self._K)
            phid = 1*np.random.gamma(100., 1./100., (len(ids[name]), self._K))
            Elogthetad = dirichlet_expectation(gammad)

            edgesd = self._A[name]
            nd = self._N[name]
            ctd = cts[name]
            
            betad = self._beta[:, ids[name]]
            temp = np.zeros(self._K)
            for ee in edgesd:
                temp = temp + self._eta * self._pi[ee] #/ nd
                    
            meanchange = 0.0
            for it in range(0, self._max_it):
                lastgamma = gammad
                
                    
                phid = np.exp(temp + Elogthetad) * betad.T
                phid = phid / (np.sum(phid, axis = 1)[:, np.newaxis] + 1e-100)
                
                gamma = self._alpha + np.sum(ctd * phid.T, axis = 1)
                Elogthetad = dirichlet_expectation(gammad)
                
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
                
            self._gamma[name] = gammad
            sstats[:, ids[name]] += ctd * phid.T
            self._phi[name] = phid
            self._pi[name] = np.sum(ctd * phid.T, axis = 1) / nd
        
        ii = 0
        for i in range(0, self._V):
            if sstats[0][i] == 0.0:
                ii = ii + 1
        
        return sstats
    
    def do_m_step(self, ids, cts):
        sstats = self.do_e_step(ids, cts)
        self._beta = sstats / np.sum(sstats, axis = 1)[:, np.newaxis]
        pi_sum = 0.0
        
        for name in self._names:
            
            for edge in self._A[name]:
                pi_sum += self._pi[name] * self._pi[edge]
              
        
        pi_sum = pi_sum / 2.0
        
        
        pi_alpha = np.zeros(self._K) + self._alpha / (self._alpha * self._K) * self._alpha / (self._alpha * self._K)
        self._nu = np.log(self._M - np.sum(pi_sum)) - np.log(self._rho * (self._K - 1) / self._K + self._M - np.sum(pi_sum))
        self._eta = np.log(pi_sum) - np.log(pi_sum + self._rho * pi_alpha) - self._nu
        elbo = self.compute_elbo(ids, cts)
        return elbo
        
    def compute_elbo(self, ids, cts):
        elbo = 0

        logbeta = np.log(self._beta + 1e-10)

        for name in self._names:
            gammad = self._gamma[name]
            id = ids[name]
            ctd = cts[name]
            Elogtheta = dirichlet_expectation(gammad)[:, np.newaxis]
            phid = self._phi[name]
            pid = self._pi[name]

            elbo += np.sum(ctd * (phid.T * logbeta[:, id]))  # E_q[log p(w_{d,n}|\beta,z_{d,n})]
            elbo += np.sum((self._alpha - 1.0) * Elogtheta)  # E_q[log p(\theta_d | alpha)]
            elbo += np.sum(phid.T * Elogtheta)  # E_q[log p(z_{d,n}|\theta_d)]

            elbo += -gammaln(np.sum(gammad)) + np.sum(gammaln(gammad)) \
                    - np.sum((gammad - 1.) * Elogtheta)  # - E_q[log q(theta|gamma)]
            elbo += - np.sum(ctd * phid.T * np.log(phid.T + 1e-20))  # - E_q[log q(z|phi)]

            for edge in self._A[name]:
                elbo += np.dot(self._eta, (pid * self._pi[edge])) + self._nu  # E_q[log p(y_{d1,d2}|z_{d1},z_{d2},\eta,\nu)]

        return elbo


        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
