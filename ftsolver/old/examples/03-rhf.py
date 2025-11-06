'''
test the RHF solver
dependency: pyscf
Chong Sun, Feb. 27, 2020
'''
import numpy as np
import numpy
import sys
from ftsolver import fted as gfted
from ftsolver import fted_spin0 as rfted

from pyscf import gto, scf, ao2mo
from functools import reduce

mol = gto.M(
    atom = [['H', (0, 0, 0)],
            ['H', (3, 0, 0)],
            ['H', (0, 3, 0)],
            ['H', (3, 3, 0)]],
    basis = 'sto6g',
    verbose = 0,
)
nelec = mol.nelectron
na = nelec//2
nb = nelec - na
nelec = (na, nb)
myhf = scf.RHF(mol).run()
c = myhf.mo_coeff
h1e = reduce(numpy.dot, (c.T, myhf.get_hcore(), c))
g2e = ao2mo.incore.full(myhf._eri, c)
norb = h1e.shape[-1]

mu0 = 0
mu = 0
beta = 1

#rdm1, rdm2, e = gfted.rdm12s_fted(h1e,g2e,norb,nelec,beta,mu0,symm='RHF')
#rdm1r, rdm2r, er = rfted.rdm12s_fted(h1e,g2e,norb,nelec,beta,mu0)
#print(np.linalg.norm(rdm1 - rdm1r))
#print(np.linalg.norm(rdm2 - rdm2r))
#print(e - er)


#e0 = gfted.energy(h1e,g2e,norb,nelec,beta,mu0,symm='RHF')
#e1 = rfted.energy(h1e,g2e,norb,nelec,beta,mu0)
#print(e1 - e0)

#ne, ge = gfted.elec_number(mu,h1e,g2e,norb,beta,symm='RHF')
#ne1, ge1 = rfted.elec_number(mu,h1e,g2e,norb,beta)
#print(ne, ne1)
#print(ge, ge1)

mu2 = gfted.solve_mu(h1e,g2e,norb,nelec,beta,mu0,symm='RHF')
mu1 = rfted.solve_mu(h1e,g2e,norb,nelec,beta,mu0,symm='RHF')

print(mu1, mu2)
