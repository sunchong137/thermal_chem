import os
import numpy as np
from pyscf import gto, scf, ao2mo
from scipy.optimize import minimize
from functools import partial

def get_mu_gc_mf(h1e, h2e, norb, nelec, spin, beta=np.inf, mu0=None,
                 max_cycle=10):
    '''
    Approximate mu_gc with mean-field solution.
    '''

    mol = gto.M()
    mol.nelectron = nelec
    mol.nao = norb
    mol.spin = spin
    mol.verbose = 0
    mol.incore_anyway = True
    mol.build()
    nelec_target = np.sum(nelec)

    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: np.eye(norb)

    ndim = len(h2e.shape)
    if ndim%2 == 1:
        if np.allclose(h2e[0], h2e[1]) and np.allclose(h2e[0], h2e[2]):
            mf._eri = ao2mo.restore(1, h2e[0], norb)
        else:
            # pyscf didn't implement get_jk for UHF h2e.
            eri = []
            for i in range(len(h2e)):
                eri.append(ao2mo.restore(1, h2e[i], norb))

            mf._eri = np.array(eri)
            mf.get_jk = partial(_get_jk_uhf, mf) #lambda *args: _get_jk_uhf( *args)
    else:
        mf._eri = ao2mo.restore(1, h2e, norb)
        
    if beta < 1e3:
        mf = scf.addons.smearing_(mf, sigma=1./beta, method='fermi')
    mf.kernel()
    mo_energy = mf.mo_energy

    
    def _fermi_smearing_occ(mu):
        '''Fermi-Dirac smearing'''
        occ = np.zeros_like(mo_energy)
        de = (mo_energy - mu) * beta
        # occ = 1./(np.exp(de)+1.)
        occ[de<400] = 1./(np.exp(de[de<400])+1.)
        return occ

    def func(mu):
        ne = np.sum(_fermi_smearing_occ(mu))
        return (ne-nelec_target)**2, 2*(ne-nelec_target)
    
    if mu0 is None:
        mu0 = mo_energy[0][nelec_target//2-1]

    res = minimize(func, mu0, method="BFGS", jac=True, \
                options={'disp':False, 'gtol':1e-3, 'maxiter':max_cycle})

    mu_n = res.x[0]
    print("Converged mu_gc from mean-field solver : mu_gc(HF) = %10.12f"%mu_n)
    return mu_n


def _get_jk_uhf(self, mols, dms, hermi=0):
    '''
    ERI has the form (aa, ab, bb).
    '''
    # from pyscf.scf import hf 
    vj_a = np.einsum('ijkl, ji -> kl', self._eri[0], dms[0]) + np.einsum('ijkl, ji -> kl', self._eri[1], dms[1])
    vj_b = np.einsum('ijkl, ji -> kl', self._eri[2], dms[1]) + np.einsum('ijkl, ji -> kl', self._eri[1], dms[0])

    vk_a = np.einsum('ijkl, kj -> il', self._eri[0], dms[0])
    vk_b = np.einsum('ijkl, kj -> il', self._eri[2], dms[1])

    vj = np.array([vj_a, vj_b])
    vk = np.array([vk_a, vk_b])
    return (vj/2, vk)  


def find_scratch_path(base='/scratch/'):
    '''
    A not so elegant way to find your scratch directory.
    Args:
        base: assuming all scratch directorys are in `/scratch/`
    '''
    if not os.path.isdir(base):
        # print(f"{base} does not exist. Using './'")
        return "./"
    
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            try:
                os.chdir(path)
                print(f"Changed scratch path to: {path}")
                return path
            except Exception as e:
                # e.g., PermissionError or OSError
                print(f"Cannot access {path}: {e}")
                continue        
    # print(f"No accessible directory found under {base}. Using './'")
    return "./"
