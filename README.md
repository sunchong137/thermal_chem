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

## Features

- Finite-temperature FCI solver.
    - Grand canonical ensemble: `gftfci.py`
    - Canonical ensemble: `cftfci.py`
- Finite-temperature DMRG solver.
    - Grand canonical, ancilla approach, `ftdmrg_wrapper.py`
    - Canonical ensemble of low-energy states, `cdmrg_wrapper.py`
Set up
------
Add the path to README to your $PYTHONPATH. 

