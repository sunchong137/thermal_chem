Finite-Temperature Quantum Chemistry Methods
============================================

A collection of finite-temperature quantum chemistry solvers.

**Authors**:

    Chong Sun <sunchong137@gmail.com>

## Dependency
All dependencies can be installed with `pip`.
- [pyscf](https://pyscf.org)
- python3
- h5py

#### Optional

- [block2](https://block2.readthedocs.io/en/latest/) for DMRG solvers.

## Install

```bash
git clone git@github.com:sunchong137/thermal_chem.git
cd thermal_chem
pip install -e .
```


## Features

- **Finite-temperature FCI solver.**
    - Grand canonical ensemble: `gftfci.py`
    - Canonical ensemble: `cftfci.py`
- **Finite-temperature DMRG solver.**
    - Grand canonical, ancilla approach, `ftdmrg_wrapper.py`
    - Canonical ensemble of low-energy states, `cdmrg_wrapper.py`


## Citing Thermal Chem
If you are using this package, we appreciate you cite the following work that involved development of this package.

```
@article{sun2020ftdmet,
  author  = {Sun, Chong and Ray, Ushinsh and Cui, Zhi-Hao and Stoudenmire, Miles and Ferrero, Michel and Chan, Garnet Kin-Lic},
  title   = {Finite-temperature density matrix embedding theory},
  journal = {Physical Review B},
  year    = {2020},
  volume  = {101},
  number  = {7},
  pages   = {075131},
  doi = {https://doi.org/10.1103/PhysRevB.101.075131}
}
```

## Acknowledgement
We thank Huanchen Zhai for helpful discussions on the Block2 package.
