#!/usr/bin/env python

"""
Canonical DMRG wrapper, not the DMET solver. (take the low excited states.)
Dependency: block2 `pip install block2`

Author:
    Chong Sun <sunchong137@gmail.com>
"""

import os
import numpy as np
import sys
import uuid
from pyblock2 import ftdmrg
from block2 import TETypes
from thermal_chem import utils
from pyscf import ao2mo
from pyblock2.driver.core import DMRGDriver, SymmetryTypes



class cDMRGSolver:
    def __init__(self, restricted=False, beta=np.inf, bmax=1e3, 
                 nroots=1, restart=False, bond_dims=[100],
                 max_memory=4000, tol=1e-8, max_cycle=20, 
                 scratch=None, noises=None, n_threads=4, thrds=None,
                 verbose=4, stdout=None, gap_buffer=1, **kwargs):
        """
        Canonical DMRG solver.
        Args:
            restricted: bool, spin symmetry
            beta: float, inverse temperature 
            bmax: float, maximum beta
            nroots: int, number of low-energy states to solve
            restart: bool, whether to restart from previous DMRG calculation 
            bond_dims: list of int, bond dimensions for each sweep
            max_memory: int, maximum stack of memory (in MB)
            tol: float, convergence tolerance
            max_cycle: int, maximum number of DMRG sweeps
            scratch: str, directory for temporary files - important for restarting, don't use the same scratch for different calculations
            noises: list of float, noise values for each sweep
            n_threads: int, number of OpenMP threads
            thrds: list of float, truncation thresholds for each sweep
            verbose: int, verbosity.
            stdout: file-like object, default sys.stdout
            gap_buffer: float, energy gap buffer for excited states
        """

        self.restricted  = restricted

        # finite-temperature arguments
        self.beta = beta 
        self.bmax = bmax

        self.norb = None
        self.nelec = None
        self.e_tot = None
        self.rdm1 = None
        self.spin = None
        self.part_func = None # partition function

        # DMRG flags
        self.nroots = nroots
        self.restart = restart
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.tol = self.conv_tol = tol
        self.n_threads = n_threads
        self.reorder_idx = None
        self.converged = None
        if isinstance(bond_dims, int):
            bond_dims = [bond_dims]
        self.bond_dims = bond_dims
        self.noises = noises
        self.thrds = thrds
        self.verbose = verbose
        if self.restricted: # RHF
            self.symm_type = SymmetryTypes.SU2
        else: # UHF
            self.symm_type = SymmetryTypes.SZ

        if scratch is None:
            scratch = utils.find_scratch_path()
        self.scratch = scratch + "/ftdmrg_scratch/" + uuid.uuid4().hex[:8] + "/"
        print("Using scratch dir: %s"%self.scratch)
        if not os.path.exists(self.scratch):
            os.makedirs(self.scratch, exist_ok=True)
            
        if stdout is None:
            self.stdout = sys.stdout
        else:
            self.stdout = stdout

        self.gap_buffer = gap_buffer
        # args used in Block2
        self.dmrg_args = {
            'scratch': scratch, 'stack_mem': self.max_memory * 1024 ** 2,
            'symm_type': self.symm_type, 'n_threads': n_threads,
            'max_cycle': self.max_cycle, 'bond_dims': self.bond_dims, 'noises': self.noises,
            'thrds': self.thrds, 'conv_tol': self.conv_tol,
            'iprint': self.verbose//2, 'restart': self.restart
        }
        
    def kernel(self, h1e, h2e, norb, nelec=None, spin=0, reorder=None, **kwargs):

        """
        Main kernel for DMRG.
        NOTE: the spin order for unrestricted H2 for dmrg is aa, ab, bb

        Args:
            h1e: one-body hamiltonian
            h2e: two-body hamiltonian
            norb: int, number of spatial orbitals
            nelec: int or tuple, number of electrons
            spin: na - nb
            reorder: reorder strategy, if None, no reorder will be applied.

        Returns:
            rdm1: density matrix (in AO representation).
            e_tot: total energy.
        """

        if nelec is None:
            nelec = norb 

        if isinstance(nelec, int):
            nelec_a = (nelec + spin) // 2
            nelec_b = (nelec - spin) // 2
        else:
            nelec_a, nelec_b = nelec

        assert (nelec_a >= 0) and (nelec_b >= 0) and (nelec_a + nelec_b == nelec)
        # self.nelec = (nelec_a, nelec_b)
        self.nelec = nelec_a + nelec_b

        h0 = 0 # core energy will be added outside
        self.norb = norb
        self.spin = spin
        beta = self.beta 
        bmax = self.bmax 
        nroots = self.nroots

        if beta >= bmax: # ground state
            raise Exception("temperature is too low, use ground state DMRG solver!")


        if not self.restricted:
            h1e = np.asarray(h1e)
            if h1e.ndim == 2:
                h1e = (h1e, h1e)
            else:
                h1e = (h1e[0], h1e[1]) # for uhf, DMRG only accepts tuple, not list or numpy array
            h2e = np.asarray(h2e) 

            if h2e.ndim == 2:
                h2e = ao2mo.restore(1, h2e, norb)
                h2e = (h2e, h2e, h2e)
            elif h2e.ndim == 3:
                h2e = (ao2mo.restore(1, h2e[0], norb),
                        ao2mo.restore(1, h2e[1], norb),
                        ao2mo.restore(1, h2e[2], norb))
            elif h2e.ndim == 4:
                h2e = (h2e, h2e, h2e)
            else:
                h2e = (h2e[0], h2e[1], h2e[2]) 

        print("Low-temp canonical DMRG solver run.")
        print("restricted = %s"%self.restricted)
        print("norb = %d"%norb)
        print("nelec = %d"%nelec)
        print("beta = %0.2f"%beta)


        if isinstance(self.bond_dims, int):
            bond_dim = [self.bond_dims]
        else:
            bond_dim = self.bond_dims

        pg = 'c1' # no group symmetry, lower case 'c', isym=1 (point group) isym ground state pg
        orb_sym = None # np.zeros(norb, dtype=np.int64) safer

        if reorder is not None:
            print("No reorder will be used in this calculation.")
        self.reorder_idx = np.array(list(range(norb)), dtype=int)

        driver = DMRGDriver(scratch=self.scratch, symm_type=self.symm_type, n_threads=self.n_threads)
        driver.initialize_system(n_sites=norb, n_elec=nelec, spin=spin, orb_sym=orb_sym)
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=h2e, ecore=0.0, iprint=0)

        # ground state calculation
        ket = driver.get_random_mps(tag="KET0", bond_dim=100, nroots=1)
        e0 = driver.dmrg(mpo, ket, n_sweeps=10, bond_dims=bond_dim, noises=[1e-5]*4 + [0],
                         thrds=[1e-8]*4 + [1e-10]*4, iprint=0) # TODO determine these parameters first.
        
        max_gap = (1+self.gap_buffer) / beta
        # derive the low-excited states
        kets = [ket]
        energies = [e0]
        for i in range(1, nroots):
            n_prev = len(kets)
            ket2 = driver.get_random_mps(tag=f"KET{i}", bond_dim=100, nroots=1)
            energy2 = driver.dmrg(mpo, ket2, n_sweeps=10, bond_dims=bond_dim, noises=[1e-5] * 4 + [0],
                    thrds=[1e-10] * 8, iprint=0, proj_weights=[5.0]*n_prev, proj_mpss=kets)
            kets.append(ket2)
            energies.append(energy2) 
            # if energy2 - e0 > max_gap:
            #     print("Energy gap exceeded the maximum allowed value. Stopping further excited state calculations.")
            #     print(f"The number of states included: {i}")
            #     self.nroots = i
            #     break

        # # canonical ensemble # TODO add shifting
        # max_exponent = 500  # to avoid overflow in exp
        # exponent0 = - e0 * beta
        # if(exponent0 > max_exponent):
        #     exp_shift = exponent0 - max_exponent
        
        # self.exp_shift = exp_shift

        part_func = np.exp(-beta * (np.array(energies) - e0))
    
        self.part_func = part_func
        E_ave = np.sum(energies * part_func) / np.sum(part_func)

        rdm1 = np.asarray(driver.get_1pdm(ket)) * part_func[0] 
        for i in range(1, self.nroots):
            rdm1 += np.asarray(driver.get_1pdm(kets[i])) * part_func[i]
        rdm1 /= np.sum(part_func)

        self.rdm1 = rdm1

        return rdm1, E_ave

    
    def make_rdm1(self):
        if (self.rdm1 is None):
            raise Exception("Call the function self.run() first to generate rdm!")
        else:
            return self.rdm1

    def make_rdm12s(self):
        '''
        DMRG twopdm order rdm2[i,j,k,l] = a^+_{i alpha} a^+_{j beta} a_{k beta} a_{l alpha}.
        '''
        rdm1 = self.rdm1 
        driver = DMRGDriver(scratch=self.scratch, symm_type=self.symm_type, n_threads=self.n_threads)
        driver.initialize_system(n_sites=self.norb, n_elec=self.nelec, spin=self.spin, orb_sym=None)
        ket = driver.load_mps(tag=f"KET0", nroots=1)
        rdm2 = np.array(driver.get_2pdm(ket)) * self.part_func[0]
        for i in range(1, self.nroots):
            ket = driver.load_mps(tag=f"KET{i}", nroots=1)
            rdm2 += np.array(driver.get_2pdm(ket)) * self.part_func[i]
        rdm2 /= np.sum(self.part_func)
        return rdm1, rdm2
    

    def clean_up(self):
        import shutil
        self.norb = None
        self.nelec = None
        self.e_tot = None
        self.rdm1 = None
        self.reorder_idx = None

        if self.scratch is not None:
             shutil.rmtree(self.scratch)

