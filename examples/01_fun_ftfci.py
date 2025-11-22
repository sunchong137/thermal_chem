import numpy as np
import sys 
from thermal_chem import gftfci

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

ftfci_solver = gftfci.gFTFCI(restricted=True)
h1e_uhf = (h1e,h1e)
h2e_uhf = (h2e, h2e, h2e)
rdm1, e = ftfci_solver.kernel(h1e_uhf, h2e_uhf, norb, nelec, beta=1/T, mu_gc=None)
print(e)
ftfci_solver.clean_up()
