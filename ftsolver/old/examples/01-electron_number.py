'''
test the calculation of electron number and the gradients.
dependency: pyscf
Chong Sun, Feb. 26, 2020
'''
import numpy as np
import numpy
import sys
from ftsolver import fted

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
eri = ao2mo.incore.full(myhf._eri, c)
norb = h1e.shape[-1]

mu0 = 0
beta = 1
ne = fted.elec_number(mu0, h1e, eri, norb, beta, symm = 'RHF')
print(ne[0])
print("gradient: ", ne[1])
#dmu = 0.01
#ne_m, _ = fted.elec_number(mu0-dmu, h1e, eri, norb, beta, symm = 'RHF')
#print ne_m
#ne_p, _ = fted.elec_number(mu0+dmu, h1e, eri, norb, beta, symm = 'RHF')
#print ne_p
#
#print (ne_p[0] - ne_m[0])/(2*dmu) 
