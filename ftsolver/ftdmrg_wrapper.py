#!/usr/bin/env python

"""
Finite temperature DMRG solver. (ancilla implementation)
Dependency: block2 `pip install block2`

Author:
    Chong Sun <sunchong137@gmail.com>
"""

import os
import numpy as np
import sys
from pyblock2 import ftdmrg
from block2 import TETypes
import ft_helpers



class FTDMRGSolver:
    def __init__(self, restricted=False, beta=np.inf, mu_gc=None,
                 restart=False, bmax=1e3, tau=0.1, bond_dims=100,
                 max_memory = 4000, tol=1e-8, max_cycle=20, 
                 scratch=None, max_bond_dim=1000, start_bond_dim=None,
                 noises=None, n_threads=4, max_mu_cycle=10,
                 verbose=4, stdout=None,  **kwargs):
        """
        FTDMRG solver
        Args:
            beta: float, inverse temperature 
            mu_gc: float, grand canonical chemical potential
            restricted: bool, spin symmetry
            bmax: float, maximum beta
            max_memory: int
                maximum stack of memory (in MB)
            verbose: 0, 1 or 2
                verbosity.
            scratch_dir: str
                Directory for writing temporary files.
        """

        self.restricted  = restricted

        # finite-temperature arguments
        self.beta = beta 
        self.mu_gc = mu_gc
        self.bmax = bmax
        self.tau = tau
        self.max_mu_cycle = max_mu_cycle
        
        
        self.cisolver = None
        
        self.scratch = scratch
        self.restart = restart
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.tol = self.conv_tol = tol

        self.norb = None
        self.nelec = None
        self.e_tot = None
        self.rdm1 = None

        # DMRG flags
        self.n_threads = n_threads
        self.reorder_idx = None
        self.converged = None
        self.bond_dims = bond_dims
        self.noises = noises
        self.max_bond_dim = max_bond_dim
        self.start_bond_dim = start_bond_dim
 
        # make scratch directory
        if self.scratch is None:
            self.scratch = ft_helpers.find_scratch_path() + "/ftdmrg_scratch/"
        if not os.path.exists(self.scratch):
            os.mkdir(self.scratch)

        if stdout is None:
            self.stdout = sys.stdout
        else:
            self.stdout = stdout

        self.verbose = verbose


    def kernel(self, h1e, h2e, norb, nelec=None, spin=0, sub_sweep_schedule=[4, 2], reorder=None, **kwargs):

        """
        Main kernel for DMRG.
        NOTE: the spin order for unrestricted H2 for dmrg is aa, ab, bb

        Args:
            h1e: one-body hamiltonian
            h2e: two-body hamiltonian
            spin: na - nb
            dm0: initial density matrix [unused, do not delete].
            mu_gc: chemical potential in GSO formalism.
            restart: whether to restart.
            nroots: include more roots for the ground state convergence.
            reorder: whether reorder orbitals during DMRG.

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
        self.nelec = (nelec_a, nelec_b)

        h0 = 0 # core energy will be added outside
        self.norb = norb
        beta = self.beta 
        mu_gc = self.mu_gc
        bmax = self.bmax 
        tau = self.tau

        if beta >= bmax: # ground state
            raise Exception("temperature is too low, use ground state DMRG solver!")

        print("restricted = %s", self.restricted)
        print("norb  = %d", norb)
        print("nelec = %d", nelec)
        print("spin  = %d", spin)
        print("beta = %0.1f"%self.beta)

        # ham = restore_ham(ham, 1, in_place=False)


        if not self.restricted:
            h1e = np.asarray(h1e)
            if h1e.ndim == 2:
                h1e = (h1e, h1e)
            else:
                h1e = (h1e[0], h1e[1]) # for uhf, DMRG only accepts tuple, not list or numpy array


        print("FTDMRG solver run.")
        print("restricted = %s"%self.restricted)
        print("norb = %d"%norb)
        print("nelec = %d", nelec)
        print("beta = %0.2f"%beta)

        # use mean-field approximated chemical potential
        mu_gc = ft_helpers.get_mu_gc_mf(np.asarray(h1e), np.asarray(h2e), norb, nelec, spin, beta, 
                                        mu0=mu_gc, max_cycle=self.max_mu_cycle)
        self.mu_gc = mu_gc

        if isinstance(self.bond_dims, int):
            bond_dim = self.bond_dims
        else:
            bond_dim = self.bond_dims[0]

        pg = 'c1' # no group symmetry, lower case 'c', isym=1 (point group) isym ground state pg
        orb_sym = np.ones(norb, dtype=np.int64) # no symmetry: 1 for each orbital
        if reorder is not None:
            print("No reorder will be used in this calculation.")
        self.reorder_idx = np.array(list(range(norb)), dtype=int)


        n_steps = int(round(beta/(2*tau)) + 0.1)  # ancilla method only evolves to beta/2       
        # clean up the existing cisolver
        if self.cisolver is not None:
            del self.cisolver
            self.cisolver = None
        
        cisolver = ftdmrg.FTDMRG(scratch=self.scratch, memory=self.max_memory*1e6, 
                                 verbose=self.verbose, omp_threads=self.n_threads) # memory in byte
        
        cisolver.init_hamiltonian(pg, n_sites=norb, twos=spin, isym=1, orb_sym=orb_sym, 
                                  e_core=h0, h1e=h1e, g2e=h2e) # twos = 2*Sz alpha-beta, 
        cisolver.generate_initial_mps(bond_dim)

        # use more sweeps for the first beta step, after the first beta step, use 2 sweeps (or 1 sweep) for each beta step
        cisolver.imaginary_time_evolution(1, tau, mu_gc, [bond_dim], TETypes.RK4, n_sub_sweeps=sub_sweep_schedule[0])
        cisolver.imaginary_time_evolution(n_steps-1, tau, mu_gc, [bond_dim], TETypes.RK4, n_sub_sweeps=sub_sweep_schedule[1], cont=True) # cont=True to start from last step

        rdm1 = np.asarray(cisolver.get_one_pdm(self.reorder_idx))
        self.rdm1 = rdm1
        
        E = 0.
        self.cisolver = cisolver # save cisolver for other usage
        return rdm1, E


    
    def make_rdm1(self):
        if (self.rdm1 is None):
            raise Exception("Call the function self.run() first to generate rdm!")
        else:
            return self.rdm1

    def make_rdm12s(self):
        '''
        DMRG twopdm order rdm2[i,j,k,l] = a^+_{i alpha} a^+_{j beta} a_{k beta} a_{l alpha}.
        '''
        if self.cisolver is None:
            raise Exception("Please call run function first to update cisolver!")
        else:
            cisolver = self.cisolver
            rdm1 = self.rdm1
            rdm2 = cisolver.get_two_pdm(self.reorder_idx)
            rdm2 = rdm2.transpose(0,1,4,2,3)
            # self.rdm2 = rdm2 # transpose to fted order
        return rdm1, rdm2

    def get_npc(self):
        ''' 
        <n_{p alpha} n_{q beta}>
        Used for double occupancy (p = q).
        '''
        
        if self.cisolver is None:
            raise Exception("Please call run function first to update cisolver!") 
        else:
            cisolver = self.cisolver
            npc1 = cisolver.get_one_npc()
            return npc1

    def get_npc_imp(self, basis, npc=None):
        if npc is None:
            npc = self.get_npc()
        nscsites = basis.shape[2]
        npc_imp = np.empty((4, nscsites, nscsites))
        npc_imp[0] = np.dot(basis[0, 0], np.dot(npc[0], basis[0, 0].T))
        npc_imp[1] = np.dot(basis[0, 0], np.dot(npc[1], basis[1, 0].T))
        npc_imp[2] = np.dot(basis[1, 0], np.dot(npc[2], basis[0, 0].T))
        npc_imp[3] = np.dot(basis[1, 0], np.dot(npc[3], basis[1, 0].T))

        return npc_imp


    def clean_up(self):
        import shutil
        self.scratch = None
        self.norb = None
        self.nelec = None
        self.e_tot = None
        self.rdm1 = None
        self.reorder_idx = None
        if self.cisolver is not None:
            del self.cisolver
            self.cisolver = None
        # shutil.rmtree(self.scratch)

