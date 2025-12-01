
import numpy as np 
from thermal_chem import ltfci 


def test_ltfci():

    norb = 6
    nelec = norb
    h1e = np.zeros((norb,norb))
    h2e = np.zeros((norb,norb,norb,norb))
    u = 4
    mu= u/2
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.
        h1e[i,(i-1)%norb] = -1.

    for i in range(norb):
        h2e[i,i,i,i] = u

    h1e = (h1e, h1e)
    h2e = (h2e*0, h2e, h2e*0)

    beta = 20.0
    delta_nelec = 1
    # Embedding system 


    restricted = False
    nroots = 5

    cisolver = ltfci.LTFCI(restricted=restricted, solve_mu=True, clean_scratch=True, 
                        nroots=nroots, delta_nelec=delta_nelec)
    rdm1, e = cisolver.kernel(h1e, h2e, norb, nelec, beta=beta, mu_gc=mu)

    cisolver.clean_up()
    assert abs(e/norb - (-0.6114509513004917)) < 1e-5
