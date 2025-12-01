'''
Low-temperature exact diagonalization (FTED).
Author: 
    Chong Sun <sunchong137@gmail.com>
'''

import numpy as np
from pyscf.fci import cistring
from pyscf.fci import direct_uhf, direct_spin1
from datetime import datetime
from functools import partial
from thermal_chem import utils
import uuid
import os
import sys
import h5py

class LTFCI:
    ''' 
    LT-FCI solver.
    '''
    def __init__(self, restricted=True, solve_mu=False, max_cycle=200, 
                 max_memory=120000, verbose=4, conv_tol=1e-10, stdout=None, 
                 scratch=None, ne_tol=5e-2, max_mu_cycle=10, clean_scratch=True,
                 max_nroots=None, max_delta_nelec=None, charge_gap_tol=None, energy_gap_tol=None,
                 **kwargs):
        '''
        Args:
            restricted: if true, spin symmetry is preserved.
            solve_mu: if true, optimize `mu_gc` to control the electron numbers.
            max_cycle: maximum iterations for the fci solver.
            max_memory: maxinum memory for the fci solver.
            verbose: higher integers means more output information is printed.
            conv_tol: convergence tolerance for the fci solver.
            stdout: output direction.
            scratch: path to save the temporary files.
            ne_tol: tolerance for the electron number when optimizing mu.
            max_mu_cycle: maximum iterations for optimizing mu.
            clean_scratch: if true, remove the scratch files after the calculation.
            max_nroots: maximal number of states to be solved in each electron number sector.
            max_delta_nelec: maximal electron number fluctuation in grand canonical per spin.
            charge_gap_tol: tolerance for including electron number fluctuations.
            energy_gap_tol: tolerance for including energy fluctuations given a electron number section.
        '''

        self.restricted = restricted 
        self.solve_mu = solve_mu
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
        self.mu_gc = None
        self.cisolver = None
        self.spin = None
        self.max_delta_nelec = max_delta_nelec 
        self.max_nroots = max_nroots
        self.charge_gap_tol = charge_gap_tol
        self.energy_gap_tol = energy_gap_tol

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
            self.scratch = utils.find_scratch_path() + "/lted_scratch/"
        if not os.path.exists(self.scratch):
            os.mkdir(self.scratch)

        self.clean_scratch = clean_scratch   
        self.saved_file = None

    def kernel(self, h1e, h2e, norb, nelec, beta=np.inf, 
               mu_gc=None, bmax=1e3, max_exponent=500, exp_shift=0, 
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
        self.mu_gc = mu_gc

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
            na_target, nb_target = nelec 
        except:
            na_target = nelec // 2
            nb_target = nelec - na_target

        self.spin = na_target - nb_target
        # non-interacting case
        if np.linalg.norm(np.asarray(h2e)) < 1e-16:
            rdm1, energy = _fermi_dirac(h1e, norb, nelec, beta, mu_gc, self.restricted)
            return rdm1, energy
        
        # Temperature too low -> ground state
        if beta > bmax: 
            e, v = cisolver.kernel(h1e, h2e, norb, nelec)
            rdm1 = cisolver.make_rdm1s(v, norb, nelec)
            return np.asarray(rdm1), e

        
        if mu_gc is None or self.solve_mu:
            # mu_gc = self.get_mu_gc_lt(h1e, h2e)
            mu_gc = utils.get_mu_gc_mf(np.array(h1e), np.array(h2e), norb, nelec, self.spin, beta=beta, mu0=self.mu_gc,
                                            max_cycle=10)
            print(f"Approximated mu_gc to be {mu_gc}")

        
        # check for overflow 
        e0_shift, _ = cisolver.kernel(h1e, h2e, norb, norb)
        exponent0 = (-e0_shift + mu_gc*norb) * beta 
        if(exponent0 > max_exponent):
            exp_shift = exponent0 - max_exponent

        self.max_exponent = max_exponent
        self.e0_shift = e0_shift
        
        # Calculating energy, rdm1, partition function
        part_func = 0.
        energy = 0.
        rdm1 = np.zeros((2, norb, norb))
        neleca_av = 0
        nelecb_av = 0

        # uhf_ham_symm = np.allclose(h1e[0], h1e[1]) and np.allclose(h2e[0], h2e[2])
        ab_symm = False
        multiplier = 1.0
        if self.restricted:
            ab_symm = True
        elif np.allclose(h1e[0], h1e[1]) and np.allclose(h2e[0], h2e[2]):
            ab_symm = True

        # ab_symm = False

        # create a file to save the result
        tag = datetime.now().strftime("%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]
        saved_file = self.scratch + f"/tmpfted_n{norb}b{beta}_{tag}.hdf5"
        self.saved_file = saved_file

        if not os.path.exists(self.saved_file):
            with h5py.File(self.saved_file, "w") as f:
                f.attrs.update(dict(Norb=norb, Nelec=nelec, beta=beta, ab_symm=ab_symm, description="Grand canonical results"))
                if ab_symm:
                    f.create_group("sectors_unequal")
                    f.create_group("sectors_equal")
                else:
                    f.create_group("sectors")
                
   
        # looping over all possible electron numbers
        if self.max_delta_nelec is None:
            na_min, na_max = 0, norb
            nb_min, nb_max = 0, norb
        else:
            na_min = max(0, na_target - self.max_delta_nelec)
            na_max = min(norb, na_target + self.max_delta_nelec)
            nb_min = max(0, nb_target - self.max_delta_nelec)
            nb_max = min(norb, nb_target + self.max_delta_nelec)

        for na in range(na_min, na_max+1):
            if ab_symm:
                multiplier = 2.0
                nb_range = range(na+1, nb_max+1)
                parent_name = "sectors_unequal"
            else:
                nb_range = range(nb_min, nb_max+1)
                parent_name = "sectors"

            for nb in nb_range:
                ne = na + nb
                e0_n = cisolver.kernel(h1e, h2e, norb, (na, nb))[0]
                
                if self.charge_gap_tol is not None:
                    charge_gap = (e0_n - e0_shift) - (ne - nelec)*mu_gc
                    if charge_gap * beta > self.charge_gap_tol:
                        continue
                if self.max_nroots is not None and self.max_nroots <=10:
                    ew, ev = cisolver.kernel(h1e, h2e, norb, (na, nb), nroots=self.max_nroots)
                    
                    if isinstance(ew, float):
                        ew = np.array([ew])
                    ev = np.asarray(ev).reshape(len(ew), -1).T
                    if self.energy_gap_tol is not None:
                        e_gaps = ew - e0_n
                        ew = ew[e_gaps * beta < self.energy_gap_tol]
                        ev = ev[:, e_gaps * beta < self.energy_gap_tol]
                else:
                    ew, ev = self._diag_ham(h1e, h2e, (na, nb))
                    ew = ew[:self.max_nroots]
                    ev = ev[:,:self.max_nroots]
                    if self.energy_gap_tol is not None:
                        e_gaps = ew - e0_n
                        ew = ew[e_gaps * beta < self.energy_gap_tol]
                        ev = ev[:, e_gaps * beta < self.energy_gap_tol]
                dim_ew = len(ew)
                # save the result to file
                with h5py.File(saved_file, "a") as f:
                    grp_name = f"{parent_name}/Na={na}_Nb={nb}"
                    if grp_name in f:
                        del f[grp_name]  # optional: overwrite existing sector
                    grp = f.create_group(grp_name)
                    grp.create_dataset("eigenvalues", data=ew, compression="gzip")
                    grp.create_dataset("eigenvectors", data=ev, compression="gzip", chunks=(dim_ew, 1))
                    grp.attrs.update(dict(Na=na, Nb=nb, dim=dim_ew, k=dim_ew))

                exponents = (-ew + mu_gc*ne) * beta
                # shift the overflow
                exponents -= exp_shift

                ndim = len(ew) 
                z_ = np.sum(np.exp(exponents))
                part_func += z_  * multiplier
                energy += np.sum(np.exp(exponents)*ew) * multiplier
                if ab_symm:
                    neleca_av += ne * z_
                    nelecb_av += ne * z_
                else:
                    neleca_av += na * z_
                    nelecb_av += nb * z_

                for i in range(ndim):
                    dm1 = cisolver.make_rdm1s(ev[:,i].copy(), norb, (na,nb))
                    dm1 = np.asarray(dm1)
                    if ab_symm:
                        rdm1[0] += (dm1[0] + dm1[1]) * np.exp(exponents[i])
                    else:
                        rdm1 += dm1 * np.exp(exponents[i])
        if ab_symm: # with ab symmetry, we didn't consider na = nb  
            rdm1[1] = np.copy(rdm1[0])    
            for na in range(na_min, na_max+1):
                nb = na
                ne = na + nb
                e0_n = cisolver.kernel(h1e, h2e, norb, (na, nb))[0]
                if self.charge_gap_tol is not None:
                    charge_gap = (e0_n - e0_shift) - (ne - nelec)*mu_gc
                    if charge_gap * beta > self.charge_gap_tol:
                        continue
                if self.max_nroots is not None and self.max_nroots <=10:
                    ew, ev = cisolver.kernel(h1e, h2e, norb, (na, nb), nroots=self.max_nroots)
                    if isinstance(ew, float):
                        ew = np.array([ew])
                    ev = np.asarray(ev).reshape(len(ew), -1).T
                    if self.energy_gap_tol is not None:
                        e_gaps = ew - e0_n
                        ew = ew[e_gaps * beta < self.energy_gap_tol]
                        ev = ev[:, e_gaps * beta < self.energy_gap_tol]
                    
                else:
                    ew, ev = self._diag_ham(h1e, h2e, (na, nb))
                    ew = ew[:self.max_nroots]
                    ev = ev[:,:self.max_nroots]
                    if self.energy_gap_tol is not None:
                        e_gaps = ew - e0_n
                        ew = ew[e_gaps * beta < self.energy_gap_tol]
                        ev = ev[:, e_gaps * beta < self.energy_gap_tol]
                dim_ew = len(ew)
                # save the result to file
                with h5py.File(saved_file, "a") as f:
                    grp_name = f"sectors_equal/Na={na}_Nb={nb}"
                    if grp_name in f:
                        del f[grp_name]  # optional: overwrite existing sector
                    grp = f.create_group(grp_name)
                    grp.create_dataset("eigenvalues", data=ew, compression="gzip")
                    grp.create_dataset("eigenvectors", data=ev, compression="gzip", chunks=(dim_ew, 1))
                    grp.attrs.update(dict(Na=na, Nb=nb, dim=dim_ew, k=dim_ew))

                exponents = (-ew + mu_gc*ne) * beta
                # shift the overflow
                exponents -= exp_shift
                
                ndim = len(ew) 
                z_ = np.sum(np.exp(exponents))
                part_func += z_
                energy += np.sum(np.exp(exponents)*ew)
                neleca_av += na * z_
                nelecb_av += nb * z_
            
                for i in range(ndim):
                    dm1 = cisolver.make_rdm1s(ev[:,i].copy(), norb, (na,nb))
                    dm1 = np.asarray(dm1)
                    rdm1 += dm1 * np.exp(exponents[i])


        energy /= part_func
        rdm1 /= part_func
        neleca_av /= part_func
        nelecb_av /= part_func

        if self.solve_mu:
            ne_diff = abs((neleca_av+nelecb_av) - (na_target+nb_target))
            if ne_diff > self.ne_tol:
                mu_gc = self.solve_mu_gc() #TODO finish
                self.mu_gc = mu_gc 
                print(f"Updated grand canonical chemical potential to: {mu_gc}")
            return self.kernel_with_checkfiles()


        print(f"Average value of electrons: Na = {neleca_av}, Nb = {nelecb_av} | Na+Nb = {neleca_av+nelecb_av}")
        print(f"energy: {energy}")

        energy = energy.real
        rdm1 = rdm1.real
        return rdm1, energy
    
    def kernel_with_checkfiles(self, exp_shift=0, **kwargs):
        '''
        Evaluate with saved energies and civectors.
        '''

        if self.mu_gc is None:
            raise ValueError("Call kernel() function first!")

        cisolver = self.cisolver 
        beta = self.beta 
        mu_gc = self.mu_gc
        norb = self.norb

        # check for overflow 
        e0_shift = self.e0_shift
        max_exponent = self.max_exponent
        exponent0 = (-e0_shift + mu_gc*norb) * beta 
        if(exponent0 > max_exponent):
            exp_shift = exponent0 - max_exponent
        

        # Calculating energy, rdm1, partition function
        part_func = 0.
        energy = 0.
        rdm1 = np.zeros((2, norb, norb))
        neleca_av = 0
        nelecb_av = 0

        # load saved file
        with h5py.File(self.saved_file, "r") as f:
            # Access individual attributes
            ab_symm = f.attrs["ab_symm"]

            if ab_symm:
                for name, grp in f["sectors_unequal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 

                    ne = na + nb
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift

                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents)) 
                    part_func += z_ * 2.0
                    energy += np.sum(np.exp(exponents)*ew) * 2.0

                    neleca_av += ne * z_
                    nelecb_av += ne * z_

                    for i in range(ndim):
                        dm1 = cisolver.make_rdm1s(ev[:,i].copy(), norb, (na,nb))
                        dm1 = np.asarray(dm1)
                        rdm1[0] += (dm1[0] + dm1[1]) * np.exp(exponents[i])

                rdm1[1] = np.copy(rdm1[0])  

                for name, grp in f["sectors_equal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    energy += np.sum(np.exp(exponents)*ew)
                    neleca_av += na * z_
                    nelecb_av += nb * z_
                
                    for i in range(ndim):
                        dm1 = cisolver.make_rdm1s(ev[:,i].copy(), norb, (na,nb))
                        dm1 = np.asarray(dm1)
                        rdm1 += dm1 * np.exp(exponents[i])

            else: # no spin symmetry
                for name, grp in f["sectors"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    energy += np.sum(np.exp(exponents)*ew)
                    neleca_av += na * z_
                    nelecb_av += nb * z_
                
                    for i in range(ndim):
                        dm1 = cisolver.make_rdm1s(ev[:,i].copy(), norb, (na,nb))
                        dm1 = np.asarray(dm1)
                        rdm1 += dm1 * np.exp(exponents[i])

        energy /= part_func
        rdm1 /= part_func
        neleca_av /= part_func
        nelecb_av /= part_func
 
        print(f"Average value of electrons: Na = {neleca_av}, Nb = {nelecb_av} | Na+Nb = {neleca_av+nelecb_av}")
        print(f"Energy evaluated: {energy}")
        energy = energy.real
        rdm1 = rdm1.real
        return rdm1, energy
    

    def make_rdm12s(self, **kwargs):
        '''
        Evaluate rdm1 and rdm2 with saved energies and civectors.
        '''

        if self.mu_gc is None:
            raise ValueError("Call kernel() function first!")

        cisolver = self.cisolver 
        beta = self.beta 
        mu_gc = self.mu_gc
        norb = self.norb

        # check for overflow 
        e0_shift = self.e0_shift
        max_exponent = self.max_exponent
        exponent0 = (-e0_shift + mu_gc*norb) * beta 
        exp_shift = 0
        if(exponent0 > max_exponent):
            exp_shift = exponent0 - max_exponent
        

        # Calculating energy, rdm1, partition function
        part_func = 0.
        # energy = 0.
        rdm1 = np.zeros((2, norb, norb))
        rdm2 = np.zeros((3, norb, norb, norb, norb))

        # load saved file
        with h5py.File(self.saved_file, "r") as f:
            # Access individual attributes
            ab_symm = f.attrs["ab_symm"]

            if ab_symm:
                for name, grp in f["sectors_unequal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 

                    ne = na + nb
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift

                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents)) 
                    part_func += z_ * 2.0
                    # energy += np.sum(np.exp(exponents)*ew) * 2.0

                    for i in range(ndim):
                        dm1, dm2 = cisolver.make_rdm12s(ev[:,i].copy(), norb, (na,nb))
                        dm1 = np.asarray(dm1)
                        dm2 = np.asarray(dm2)
                        rdm1[0] += (dm1[0] + dm1[1]) * np.exp(exponents[i])
                        rdm2[0] += (dm2[0] + dm2[2]) * np.exp(exponents[i])
                        rdm2[1] += (dm2[1] + np.transpose(dm2[1], (2,3,0,1))) * np.exp(exponents[i])

                rdm1[1] = np.copy(rdm1[0])  
                rdm2[2] = np.copy(rdm2[0])

                for name, grp in f["sectors_equal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    # energy += np.sum(np.exp(exponents)*ew)

                    for i in range(ndim):
                        dm1, dm2 = cisolver.make_rdm12s(ev[:,i].copy(), norb, (na,nb))
                        rdm1 += np.asarray(dm1) * np.exp(exponents[i]) 
                        rdm2 += np.asarray(dm2) * np.exp(exponents[i]) 

            else: # no spin symmetry
                for name, grp in f["sectors"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    # energy += np.sum(np.exp(exponents)*ew)

                    for i in range(ndim):
                        dm1, dm2 = cisolver.make_rdm12s(ev[:,i].copy(), norb, (na,nb))
                        rdm1 += np.asarray(dm1) * np.exp(exponents[i]) 
                        rdm2 += np.asarray(dm2) * np.exp(exponents[i]) 

        # energy /= part_func
        rdm1 /= part_func
        rdm2 /= part_func

        # energy = energy.real
        rdm1 = rdm1.real
        rdm2 = rdm2.real
        return rdm1, rdm2


    def eval_elec_number(self, mu_gc=None, exp_shift=0, beta=None, **kwargs):
        '''
        Evaluate the electron number.
        Returns:
            electron numbers and the gradient wrt to mu.
        '''

        if mu_gc is None:
            mu_gc = self.mu_gc

        if beta is None:    
            beta = self.beta 
        norb = self.norb

        # check for overflow 
        e0_shift = self.e0_shift
        max_exponent = self.max_exponent
        exponent0 = (-e0_shift + mu_gc*norb) * beta 
        if(exponent0 > max_exponent):
            exp_shift = exponent0 - max_exponent
        

        # Calculating energy, rdm1, partition function
        part_func = 0.
        neleca_av = 0
        nelecb_av = 0
        Ncorr_a = 0
        Ncorr_b = 0

        # load saved file
        with h5py.File(self.saved_file, "r") as f:
            # Access individual attributes
            ab_symm = f.attrs["ab_symm"]

            if ab_symm:
                for name, grp in f["sectors_unequal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ew = grp["eigenvalues"][:]

                    ne = na + nb
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift

                    z_ = np.sum(np.exp(exponents)) 
                    part_func += z_ * 2.0
                    neleca_av += ne * z_
                    nelecb_av += ne * z_
                    Ncorr_a += (ne*ne) * z_
                    Ncorr_b += (ne*ne) * z_


                for name, grp in f["sectors_equal"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    neleca_av += na * z_
                    nelecb_av += nb * z_
                    Ncorr_a += (na*ne) * np.sum(np.exp(exponents))
                    Ncorr_b += (nb*ne) * np.sum(np.exp(exponents))

            else: # no spin symmetry
                for name, grp in f["sectors"].items():
                    na, nb = grp.attrs["Na"], grp.attrs["Nb"]
                    ne = na + nb
                    ew = grp["eigenvalues"][:]
                    ev = grp["eigenvectors"][:] 
                    exponents = (-ew + mu_gc*ne) * beta
                    # shift the overflow
                    exponents -= exp_shift
                    
                    ndim = len(ew) 
                    z_ = np.sum(np.exp(exponents))
                    part_func += z_
                    neleca_av += na * z_
                    nelecb_av += nb * z_
                    Ncorr_a += (na*ne) * z_
                    Ncorr_b += (nb*ne) * z_

        neleca_av /= part_func
        nelecb_av /= part_func
        Ncorr_a /= part_func
        Ncorr_b /= part_func

        Ne = neleca_av + nelecb_av

        grad_a = beta * (Ncorr_a - neleca_av*Ne)
        grad_b = beta * (Ncorr_b - nelecb_av*Ne)
 
        # print(f"Average value of electrons: Na = {neleca_av}, Nb = {nelecb_av}")
        # print(f"Energy evaluated: {energy}")
      
        return (neleca_av, nelecb_av), (grad_a, grad_b)
    
    def solve_mu_gc(self):
        '''
        Solve for the mu value to reach the target electron number.
        '''
        from scipy.optimize import minimize

        fun_dict = {}
        jac_dict = {}

        nelec = self.nelec
        if type(nelec) != tuple:
            na = nelec//2
            nb = nelec - na 
            nelec = (na, nb)
        
        def func(x):
            mu_gc = x[0]
            if mu_gc in fun_dict:
                return fun_dict[mu_gc]
            else:
                Ne, grad = self.eval_elec_number(mu_gc)
                da = Ne[0] - nelec[0]
                db = Ne[1] - nelec[1]
                diff = da**2 + db**2
                jac = 2*da*grad[0] + 2*db*grad[1]
                jac_dict[mu_gc] = jac
                fun_dict[mu_gc] = diff
                return diff, jac

        mu0 = self.mu_gc
        res = minimize(func, mu0, method="BFGS", jac=True, \
                    options={'disp':False, 'gtol':self.ne_tol, 'maxiter':self.max_mu_cycle})

        mu_n = res.x[0]
        print("Converged mu_gc for ED solver: mu_gc(ED) = %10.12f"%mu_n)
        return mu_n

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
    

    def get_mu_gc_lt(self, h1e, h2e):
        '''
        Get the approximation of the low-temperature chemical potential.
        Mid-gap approximation.
        '''
        norb = self.norb 
        nelec = self.nelec
        if type(nelec) is tuple:
            ne = nelec[0] + nelec[1]
        else:
            ne = nelec 
            
        Ep1, _ = self.cisolver.kernel(h1e, h2e, norb, ne+1)
        Em1, _ = self.cisolver.kernel(h1e, h2e, norb, ne-1)

        mu_gc = (Ep1 - Em1)/2
        self.mu_gc = mu_gc

        return mu_gc
    
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

################# move to helpers ################# 
def _fermi_dirac(h1e, norb, nelec, beta, mu_gc, restricted=False, bmax=1e3):
    ''' 
    Non-interacting case.
    '''
    def fermi(e):
        return 1./(1.+np.exp((e-mu_gc)*beta))
    
    if restricted:
        assert nelec % 2 == 0
        neleca = nelec // 2 
        ew_a, ev_a = np.linalg.eigh(h1e)
        if beta > bmax:
            eocc_a = np.ones(norb)
            eocc_a[neleca:] = 0
        else:
            eocc_a = fermi(ew_a)
        e = 2 * np.sum(ew_a*eocc_a) 
        dm1_a = np.asarray(np.dot(ev_a, np.dot(np.diag(eocc_a), ev_a.T.conj())))
        rdm1 = np.array([dm1_a, dm1_a])

    else: # unrestricted
        if isinstance(nelec, (int, np.integer)):
            nelecb = nelec // 2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        
        h1e = np.asarray(h1e)
        if h1e.ndim > 2:
            ew_a, ev_a = np.linalg.eigh(h1e[0])
            ew_b, ev_b = np.linalg.eigh(h1e[1])
        else:
            ew_a, ev_a =  np.linalg.eigh(h1e)
            ew_b = ew_a 
            ev_b = ev_a 

        if beta > bmax:
            eocc_a, eocc_b = np.ones(norb), np.ones(norb)
            eocc_a[neleca:] = 0
            eocc_b[nelecb:] = 0
        else:
            eocc_a = fermi(ew_a)
            eocc_b = fermi(ew_b)

        e = np.sum(ew_a*eocc_a) + np.sum(ew_b*eocc_b)
        dm1_a = np.asarray(np.dot(ev_a, np.dot(np.diag(eocc_a), ev_a.T.conj())))
        dm1_b = np.asarray(np.dot(ev_b, np.dot(np.diag(eocc_b), ev_b.T.conj())))
        rdm1 = np.array([dm1_a, dm1_b])

    return rdm1, e


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

