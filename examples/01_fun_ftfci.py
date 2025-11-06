import numpy as np
import sys 
sys.path.append("../ftsolver/")
import ftfci

norb = 4
nelec = norb
h1e = np.zeros((norb,norb))
h2e = np.zeros((norb,norb,norb,norb))
T = 10
u = 4
mu= u/2
#mu= 0.0
for i in range(norb):
    h1e[i,(i+1)%norb] = -1.
    h1e[i,(i-1)%norb] = -1.

for i in range(norb):
    h2e[i,i,i,i] = u

ftfci_solver = ftfci.FTFCI(restricted=True)
h1e_uhf = (h1e,h1e)
h2e_uhf = (h2e, h2e, h2e)
rdm1, e = ftfci_solver.kernel(h1e_uhf, h2e_uhf, norb, nelec, beta=1/T, mu_gc=None)
print(rdm1)
exit()
# rdm1_p, e_p = ftfci_solver.kernel_parallel(h1e_uhf, h2e_uhf, norb, nelec, beta=1/T, mu=mu)
assert abs(e/norb - (-0.5256871208654883)) < 1e-5
assert np.allclose(np.diag(rdm1[0]), np.ones(4)*0.5)
assert np.allclose(np.diag(rdm1[1]), np.ones(4)*0.5)
# assert abs(e_p/norb - (-0.5256871208654883)) < 1e-5

rdm1, rdm2 = ftfci_solver.make_rdm12s()
assert np.allclose(np.diag(rdm1[0]), np.ones(4)*0.5)
assert np.allclose(np.diag(rdm1[1]), np.ones(4)*0.5)