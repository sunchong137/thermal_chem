
from pyscf.fci import cistring
from pyscf import scf, ao2mo
from functools import reduce
import numpy as np
import scipy as sp

def run_scf(mol, symm):
    if symm is 'RHF':
        m = scf.RHF(mol).run()
        mo_coeff = m.mo_coeff
        norb = mo_coeff.shape[-1]
        h1e = reduce(np.dot, (mo_coeff.T, m.get_hcore(), mo_coeff))
        g2e = ao2mo.kernel(mol, mo_coeff)
        norb = h1e.shape[-1]
        return h1e, g2e, norb
    elif symm is 'UHF':
        m = scf.UHF(mol).run()
        mo_coeff = m.mo_coeff
        norb = mo_coeff[0].shape[-1]
        h1ea = reduce(np.dot, (mo_coeff[0].T, m.get_hcore(), mo_coeff[0]))
        h1eb = reduce(np.dot, (mo_coeff[1].T, m.get_hcore(), mo_coeff[1]))
        g2eaa = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff[0]), norb)
        g2ebb = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff[1]), norb)
        g2eab = ao2mo.kernel(mol, [mo_coeff[0]] * 2 + [mo_coeff[1]] * 2)
        return (h1ea, h1eb), (g2eaa, g2eab, g2ebb), norb

def eigvals(h1e, g2e, norb, nelec, fcisolver):
    '''Exactly diagonalize the hamiltonian.'''
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, 0.5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na * nb

    eyemat = np.eye(ndim)
    Hmat = []
    for i in range(ndim):
        hc = fcisolver.contract_2e(h2e, eyemat[i], norb, nelec).reshape(-1)
        Hmat.append(hc)

    Hmat = np.asarray(Hmat).T
    ew = np.linalg.eigvalsh(Hmat)
    return ew

def gc_ensemble_eigs(h1e, g2e, norb, symm='UHF'):
    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'UEG':
        from pyscf.fci import direct_nosym as fcisolver
        fcisolver = fcisolver.FCISolver()
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    
    ref = []
    for na in range(0, norb + 1):
        for nb in range(na if symm is 'RHF' or symm is 'UEG' else 0, norb + 1):
            print("\r%3d%% " % ((na * norb + nb) * 100 // ((norb + 1) * norb)), end='', flush=True)
            ew = eigvals(h1e, g2e, norb, (na, nb), fcisolver)
            ref.append((na, nb, ew))
    print("\r", end='', flush=True)
    return ref

def expectation(ref, mu, beta, symm):
    def f(fn=None, fe=None):
        Z, X = 0, 0
        E0 = ref[0][2][0]
        assert (fn is None) ^ (fe is None)
        for na, nb, E in ref:
            E = np.array(E, dtype=np.float128)
            ex = np.exp(-(E - E0 - mu * (na + nb)) * beta)
            z = np.sum(ex)
            if fn is not None:
                x = fn(na, nb) * z
            elif fe is not None:
                x = np.sum(fe(E) * ex)
            if (symm is 'RHF' or symm is 'UEG') and na != nb:
                z, x = z * 2, x * 2
            Z += z
            X += x
        return X / Z
    return f

def elec_number(ref, mu, beta, symm):
    expect = expectation(ref, mu, beta, symm=symm)
    if symm == 'RHF' or symm is 'UEG':
        N  = expect(fn=lambda na, nb: na + nb)
        NN = expect(fn=lambda na, nb: (na + nb) ** 2)
        DN = beta * (NN - N * N)
        return N, DN
    elif symm == 'UHF':
        NA  = expect(fn=lambda na, _: na)
        NB  = expect(fn=lambda _, nb: nb)
        NNA = expect(fn=lambda na, _: na ** 2)
        NNB = expect(fn=lambda _, nb: nb ** 2)
        NAB = expect(fn=lambda na, nb: na * nb)
        DNA = beta * (NNA + NAB - NA * NA - NA * NB)
        DNB = beta * (NNB + NAB - NB * NB - NA * NB)
        return (NA, NB), (DNA, DNB)

def energy(ref, mu, beta, symm='UHF'):
    expect = expectation(ref, mu, beta, symm=symm)
    return expect(fe=lambda _:_)

def solve_mu(ref, mu0, beta, nelec, symm):
    def solve(mu):
        if symm is 'RHF' or symm is 'UEG':
            n, dn = elec_number(ref, mu, beta, symm)
            diff = n - nelec
            return diff ** 2, 2 * diff * dn
        elif symm is 'UHF':
            (na, nb), (dna, dnb) = elec_number(ref, mu, beta, symm)
            diffa, diffb = na - nelec[0], nb - nelec[1]
            return diffa ** 2 + diffb ** 2, 2 * diffa * dna + 2 * diffb * dnb
    mu_cache = {}
    def cached(mu):
        mu = mu[0]
        if mu not in mu_cache:
            mu_cache[mu] = solve(mu)
        return mu_cache[mu]
    f = lambda mu: cached(mu)[0]
    g = lambda mu: cached(mu)[1]
    opt_mu = sp.optimize.minimize(f, mu0, method="CG", jac=g,
        options={'disp': False, 'gtol': 1e-6, 'maxiter': 10})
    return opt_mu.x[0]
