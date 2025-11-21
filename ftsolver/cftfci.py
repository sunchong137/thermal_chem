'''
Canonical finite-temperature exact diagonalization (FTED).

Author: 
    Chong Sun <sunchong137@gmail.com>
'''

import numpy as np
from pyscf.fci import cistring
from pyscf.fci import direct_uhf, direct_spin1
from datetime import datetime
from functools import partial
import uuid
import os
import sys
import h5py

class cFTFCI:
    ''' 
    FT-FCI solver.
    '''
    def __init__(self, restricted=True, max_cycle=200, 
                 max_memory=120000, verbose=4, conv_tol=1e-10, stdout=None, 
                 nroots=None, 
                 scratch=None, ne_tol=5e-2, max_mu_cycle=10, clean_scratch=False,
                 **kwargs):
        '''
        Args:
            restricted: if true, spin symmetry is preserved.
            max_cycle: maximum iterations for the fci solver.
            max_memory: maxinum memory for the fci solver.
            verbose: higher integers means more output information is printed.
            conv_tol: convergence tolerance for the fci solver.
            stdout: output direction.
            nroots: number of low-energy states to solve.
            scratch: path to save the temporary files.
        '''

        self.restricted = restricted 
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = conv_tol
        self.verbose = verbose

        self.part_func = None
        self.rdm1 = None 
        self.rdm2 = None

        self.norb = None
        self.nelec = None
        self.beta = None
        self.bmax = None
        self.cisolver = None
        self.spin = None
        self.nroots = nroots

        # optimizing mu
        self.ne_tol = ne_tol
        self.max_mu_cycle = max_mu_cycle

        # overflow settings
        self.max_exponent = None 
        self.e0_shift = None 

        if stdout is None:
            self.stdout = sys.stdout
        else:
            self.stdout = stdout

        # make scratch directory
        self.scratch = scratch
        if scratch is None:
            self.scratch = find_scratch_path() + "/fted_scratch/"
        if not os.path.exists(self.scratch):
            os.mkdir(self.scratch)

        self.clean_scratch = clean_scratch   
        self.saved_file = None

    def kernel(self, h1e, h2e, norb, nelec, beta=np.inf, 
               bmax=1e3, max_exponent=500, exp_shift=0, 
               **kwargs):
        '''
        Return the expectation values of energy and rdm1 at temperature T.
        Spin symmetry unrestricted.
        Args:
            h1e: 1-body Hamiltonian in PySCF convention
            h2e: 2-body Hamiltonian in PySCF convention 
            norb: number of orbitals
            nelec: number of electrons
            beta: inverse temperature 1/T
            mu_gc: chemical potetial
            bmax: maximum beta value 
            max_exponent: the maximum allowed exponent value
        Returns:
            rdm1, energy
        '''

        self.beta = beta 
        self.norb = norb 
        self.nelec = nelec 
        self.bmax = bmax

        if self.restricted:
            cisolver = direct_spin1 
        else:
            cisolver = direct_uhf 
        cisolver.max_memory = self.max_memory
        cisolver.max_cycle = self.max_cycle
        cisolver.conv_tol = self.conv_tol
        cisolver.verbose = self.verbose
        self.cisolver = cisolver

        h1e = np.asarray(h1e)
        h2e = np.asarray(h2e)

        if self.restricted:
            if h1e.ndim > 2:
                h1e =  h1e[0]
            if h2e.ndim > 4:
                h2e = h2e[1]

        try:
            na, nb = nelec 
        except:
            na = nelec // 2
            nb = nelec - na

        self.spin = na - nb
        # non-interacting case
        
        # Temperature too low -> ground state
        if beta > bmax: 
            ew, ev = cisolver.kernel(h1e, h2e, norb, nelec)
            ew = np.array([ew])
            ev = np.asarray(ev).reshape(len(ew), -1).T
            rdm1 = np.asarray(cisolver.make_rdm1s(ev, norb, nelec))
            E_av = ew[0]

        else:
            if self.nroots is not None:
                ew, ev = cisolver.kernel(h1e, h2e, norb, nelec, nroots=self.nroots)
                ev = np.asarray(ev).reshape(len(ew), -1).T
                
                part_func = np.exp(-beta * (ew - ew[0]))
                self.part_func = part_func
                E_av = np.sum(ew * part_func) / np.sum(part_func) 
                rdm1 = np.asarray(cisolver.make_rdm1s(ev[:, 0], norb, nelec)) * part_func[0]
                for i in range(1, len(ew)):
                    rdm1 += np.asarray(cisolver.make_rdm1s(ev[:, i], norb, nelec)) * part_func[i]
                rdm1 /= np.sum(part_func)
                self.rdm1 = rdm1
            else:
                ew, ev = self._diag_ham(h1e, h2e, (na, nb))
                part_func = np.exp(-beta * (ew - ew[0]))
                self.part_func = part_func
                E_av = np.sum(ew * part_func) / np.sum(part_func) 
                rdm1 = np.asarray(cisolver.make_rdm1s(ev[:,0], norb, nelec)) * part_func[0]
                for i in range(1, len(ew)):
                    rdm1 += np.asarray(cisolver.make_rdm1s(ev[:,i], norb, nelec)) * part_func[i]
                rdm1 /= np.sum(part_func)
                self.rdm1 = rdm1
        
        # create a file to save the result
        tag = datetime.now().strftime("%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]
        saved_file = self.scratch + f"/tmpfted_n{norb}b{beta}_{tag}.hdf5"
        self.saved_file = saved_file

        # save eigenvectors and eigenvalues
        if not os.path.exists(self.saved_file):
            with h5py.File(self.saved_file, "w") as f:
                f.attrs.update(dict(Norb=norb, Nelec=nelec, beta=beta, description="Canonical results"))
                grp = f.create_group("sector")
                dim_ew = len(ew)
                grp.create_dataset("eigenvalues", data=ew, compression="gzip")
                grp.create_dataset("eigenvectors", data=ev, compression="gzip", chunks=(dim_ew, 1))
                grp.attrs.update(dict(Na=na, Nb=nb, dim=dim_ew, k=dim_ew))
        
        return np.asarray(rdm1), E_av

    
    def make_rdm12s(self, **kwargs):
        '''
        Evaluate rdm1 and rdm2 with saved energies and civectors.
        '''
        if not os.path.exists(self.saved_file):
            raise FileNotFoundError(f"{self.saved_file} does not exist.")

        cisolver = self.cisolver 
        beta = self.beta 
        norb = self.norb

        # # check for overflow 
        # e0_shift = self.e0_shift
        # max_exponent = self.max_exponent
        # exponent0 = (-e0_shift + mu_gc*norb) * beta 
        # exp_shift = 0
        # if(exponent0 > max_exponent):
        #     exp_shift = exponent0 - max_exponent
        

        rdm1 = self.rdm1

        # load saved file
        with h5py.File(self.saved_file, "r") as f:
            grp = f["sector"]
            ew = grp["eigenvalues"][:]      # shape: (dim,)
            ev = grp["eigenvectors"][:] 
            if self.beta > self.bmax:
                rdm1, rdm2 = self.cisolver.make_rdm12s(ev, norb, self.nelec)
                return rdm1, rdm2
        dm1_, dm2_ = self.cisolver.make_rdm12s(ev[:,0], norb, self.nelec)
        # rdm1 = np.asarray(dm1_) * self.part_func[0]
        rdm2 = np.asarray(dm2_) * self.part_func[0]

        for i in range(1, len(ew)):
            dm1_, dm2_ = self.cisolver.make_rdm12s(ev[:,i], norb, self.nelec)
            # rdm1 += np.asarray(dm1_) * self.part_func[i]
            rdm2 += np.asarray(dm2_) * self.part_func[i]


        # rdm1 /= np.sum(self.part_func)
        rdm2 /= np.sum(self.part_func)

        # energy = energy.real
        # rdm1 = rdm1.real
        rdm2 = rdm2.real
        return rdm1, rdm2


    def _diag_ham(self, h1e, h2e, nelec):
        '''
            Exactly diagonalize the hamiltonian.

        '''
        norb = self.norb 
        cisolver = self.cisolver

        h2e = cisolver.absorb_h1e(h1e, h2e, norb, nelec, 0.5)
        neleca, nelecb = nelec

        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
        ndim = na*nb

        eyemat = np.eye(ndim)
        hmat = [] 
        for i in range(ndim):
            hc = cisolver.contract_2e(h2e, eyemat[i], norb, nelec).reshape(-1)
            hmat.append(hc)

        hmat = np.asarray(hmat).T
        ew, ev = np.linalg.eigh(hmat)
        return ew, ev
    
    
    def clean_up(self):
        self.cisolver = None
        self.norb = None
        self.nelec = None
        self.e_tot = None
        self.rdm1 = None
        
        if self.clean_scratch:
            os.remove(self.saved_file)
            # import shutil
            # shutil.rmtree(self.saved_file)


def find_scratch_path(base='/scratch/'):
    '''
    A not so elegant way to find your scratch directory.
    Args:
        base: assuming all scratch directorys are in `/scratch/`
    '''
    if not os.path.isdir(base):
        # print(f"{base} does not exist.")
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
    print(f"No accessible directory found under {base}. Using './'")
    return "./"

