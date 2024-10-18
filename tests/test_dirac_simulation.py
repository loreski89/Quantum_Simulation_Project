
import pytest
import numpy as np
from src.dirac_simulation import DiracSimulationGPU

def test_wavefunction_initialization():
    sim = DiracSimulationGPU(mass=1.0, grid_size=100, dt=0.01, timesteps=100)
    sim.initialize_wavefunction('gaussian')
    assert sim.psi.shape == (2, 100)
    assert np.allclose(sim.psi[0], np.exp(-sim.x**2))

def test_hamiltonian_building():
    sim = DiracSimulationGPU(mass=1.0, grid_size=100, dt=0.01, timesteps=100)
    sim.build_hamiltonian('harmonic')
    assert sim.H is not None
