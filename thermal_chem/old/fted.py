#   Copyright 2016-2023 Chong Sun
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


'''
exact diagonalization solver with grand canonical statistics.
'''

import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
from scipy.optimize import minimize
from ftsolver.utils import logger as log

import datetime
import scipy
import sys
import os

from ftsolver import fted_spin0 as spin0

def rdm12s_fted(h1e, g2e, norb, nelec, beta, mu=0.0, symm='UHF', bmax=1e3, \
                dcompl=False, **kwargs):

    '''
    Return the expectation values of energy, RDM1 and RDM2 at temperature T.
    '''

    if symm is 'RHF':
        return spin0.rdm12s_fted(h1e,g2e,norb,nelec,beta,mu,bmax)
        #from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    

    Z = 0.
    E = 0.
    RDM1 = np.zeros((2, norb, norb), dtype=np.complex128)
    RDM2 = np.zeros((3, norb, norb, norb, norb), dtype=np.complex128)

    # non-interacting case
    if np.linalg.norm(g2e[1]) == 0:
        RDM1, E = FD(h1e,norb,nelec,beta,mu,symm)
        return RDM1, RDM2, E

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if beta > bmax:
        e, v = fcisolver.kernel(h1e, g2e, norb, nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        return np.asarray(RDM1), np.asarray(RDM2), e


    # check for overflow
    e0, _ = fcisolver.kernel(h1e,g2e,norb,norb)
    exp_max = (-e0+mu*norb)*beta
    if(exp_max > 700):
        exp_shift = exp_max - 500
    else:
        exp_shift = 0

    # Calculating E, RDM1, Z
    Na = 0
    Nb = 0
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            # ne = na + nb
            ew, ev = diagH(h1e,g2e,norb,(na,nb),fcisolver)
            exp_ = (-ew+mu*(na+nb))*beta
            exp_ -= exp_shift
            ndim = len(ew) 
            Z += np.sum(np.exp(exp_))
            E += np.sum(np.exp(exp_)*ew)
            Na += na * np.sum(np.exp(exp_))
            Nb += nb * np.sum(np.exp(exp_))
         
            for i in range(ndim):
                dm1, dm2 = fcisolver.make_rdm12s(ev[:,i].copy(),norb,(na,nb))
                dm1 = np.asarray(dm1,dtype=np.complex128)
                dm2 = np.asarray(dm2,dtype=np.complex128)
                RDM1 += dm1*np.exp(exp_[i])
                RDM2 += dm2*np.exp(exp_[i])

    E /= Z
    Na /= Z
    Nb /= Z
    RDM1 /= Z
    RDM2 /= Z

    #print "%.6f        %.6f"%(1./T, N.real)
    log.result("The number of electrons in embedding space: " )
    log.result("N(alpha) = %10.10f"%Na)
    log.result("N(beta)  = %10.10f"%Nb)

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real
    # RDM2 order: aaaa, aabb, bbbb
    log.result("FTED energy: %10.12f"%E)
    return RDM1, RDM2, E

def energy(h1e, g2e, norb, nelec, beta, mu=0.0, symm='UHF', bmax=1e3, \
                dcompl=False,**kwargs):

    if symm is 'RHF':
        return spin0.energy(h1e,g2e,norb,nelec,beta,mu,bmax)
        #from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    
    Z = 0.
    E = 0.

    if np.linalg.norm(g2e[1]) == 0:
        _, E = FD(h1e,norb,nelec,beta,mu,symm)
        return E

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if beta > bmax:
        e, _ = fcisolver.kernel(h1e, g2e, norb, nelec)
        return e

    # check for overflow
    e0, _ = fcisolver.kernel(h1e,g2e,norb,norb)
    exp_max = (-e0+mu*norb)*beta
    if(exp_max > 700):
        exp_shift = exp_max - 500
    else:
        exp_shift = 0

    # Calculating E, RDM1, Z
    N = 0
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            ne = na + nb
            ew, _ = diagH(h1e,g2e,norb,(na,nb),fcisolver)
            exp_ = (-ew+mu*(na+nb))*beta
            exp_ -= exp_shift
            # ndim = len(ew) 
            Z += np.sum(np.exp(exp_))
            E += np.sum(np.exp(exp_)*ew)
            N += ne * np.sum(np.exp(exp_))
         
    E /= Z
    N /= Z

    if not dcompl:
        E = E.real
    return E


def elec_number(mu, h1e, g2e, norb, beta, symm='UHF', bmax=1e3, \
                dcompl=False, **kwargs):
    '''
        return:
                electron number in form of (Na, Nb)
                gradient wrt mu (dNa/dmu, dNb/dmu)
    '''
    if symm is 'RHF':
        return spin0.elec_number(mu,h1e,g2e,norb,beta,bmax)
        #from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver 

    Z = 0. 
    Na = 0
    Nb = 0
    Ncorr_a = 0
    Ncorr_b = 0
    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    # check for overflow
    e0, _ = fcisolver.kernel(h1e,g2e,norb,norb)
    exp_max = (-e0+mu*norb)*beta
    if(exp_max > 700):
        exp_shift = exp_max - 500
    else:
        exp_shift = 0

    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            ne = na + nb
            ew, ev = diagH(h1e,g2e,norb,(na,nb),fcisolver)
            exp_ = (-ew+mu*(na+nb))*beta
            exp_ -= exp_shift
            # ndim = len(ew)
            Z += np.sum(np.exp(exp_))
            Na += na * np.sum(np.exp(exp_))
            Nb += nb * np.sum(np.exp(exp_))
            Ncorr_a += (na*ne) * np.sum(np.exp(exp_))
            Ncorr_b += (nb*ne) * np.sum(np.exp(exp_))
        

    Na /= Z 
    Nb /= Z 
    Ncorr_a /= Z
    Ncorr_b /= Z

    N = Na + Nb
    
    grad_a = beta * (Ncorr_a - Na*N)
    grad_b = beta * (Ncorr_b - Nb*N)

    return (Na, Nb), (grad_a, grad_b)

def solve_mu(h1e, g2e, norb, nelec, beta, mu0=0.0, symm='UHF', bmax=1e3, \
                dcompl=False,**kwargs):

    '''
    fit mu to match the given electon number.
    using: CG
    '''

    if (symm == 'RHF'):
        return spin0.solve_mu(h1e,g2e,norb,nelec,beta,mu0,bmax)

    if beta > bmax:
        return mu0

    fun_dict = {}
    jac_dict = {}
    
    def func(x):
        mu = x[0]
        if mu in fun_dict:
            return fun_dict[mu]
        else:
            Ne, grad = elec_number(mu,h1e,g2e,norb,beta,symm)
            da = Ne[0] - nelec[0]
            db = Ne[1] - nelec[1]
            diff = da**2 + db**2
            jac = 2 * da * grad[0] + 2 * db * grad[1]
            
            jac_dict[mu] = jac
            fun_dict[mu] = diff
            return diff

    def grad(x):
        mu = x[0]
        if mu in jac_dict:
            return jac_dict[mu]
        else:
            Ne, grad = elec_number(mu,h1e,g2e,norb,beta,symm)
            da = Ne[0] - nelec[0]
            db = Ne[1] - nelec[1]
            diff = da**2 + db**2
            jac = 2 * da * grad[0] + 2 * db * grad[1]
            
            jac_dict[mu] = jac
            fun_dict[mu] = diff
            return jac

    res = minimize(func, mu0, method="CG", jac=grad, \
                   options={'disp':True, 'gtol':1e-4, 'maxiter':10})

    mu_n = res.x[0]
    print("Converged mu for ED solver: mu(ED) = %10.12f"%mu_n)
    return mu_n


def diagH(h1e, g2e, norb, nelec, fcisolver):
    '''
        exactly diagonalize the hamiltonian.
    '''
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na*nb

    eyemat = np.eye(ndim)
    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyemat[i])
        Hmat.append(hc)
    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev


######################################################################
def FD(h1e,norb,nelec,beta,mu,symm='RHF'):
    #ew, ev = np.linalg.eigh(h1e)
    htot = np.zeros((2*norb, 2*norb))
    htot[:norb,:norb] = h1e[0]
    htot[norb:,norb:] = h1e[1]
            
    ew, ev = np.linalg.eigh(htot)
    def fermi(mu):
        return 1./(1.+np.exp((ew-mu)*beta))
    if beta > 1e3:
        beta = np.inf
        eocc = np.ones(2*norb)
        eocc[nelec:]*=0.
    else:
        eocc = fermi(mu)

    dm1 = np.asarray(np.dot(ev, np.dot(np.diag(eocc), ev.T.conj())), dtype=np.complex128)
    e = np.sum(ew*eocc)
    DM1 = (dm1[:norb, :norb], dm1[norb:, norb:])

    return DM1, e

######################################################################
if __name__ == '__main__':

    import sys
    norb = 4
    nimp = 2
    nelec = 4
    h1e = np.zeros((norb,norb))
    g2e = np.zeros((norb,norb,norb,norb))
    #T = 0.02
    u = float(sys.argv[1])
    mu= u/2
    #mu= 0.0
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.
        h1e[i,(i-1)%norb] = -1.
    #h1e[0,-1] = 0.
    #h1e[-1,0] = 0.

    #for i in range(norb):
    #    h1e[i,i] += -u/2.

    for i in range(norb):
        g2e[i,i,i,i] = u

    T = 0.01
    dm1,_,e1 = rdm12s_fted((h1e,h1e),(g2e*0, g2e, g2e*0),norb,nelec,T,mu, symm='UHF')
    print(e1/norb)#+u/2.)
