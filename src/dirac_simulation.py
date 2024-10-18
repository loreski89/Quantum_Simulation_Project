
import numpy as np
from numba import cuda

@cuda.jit
def evolve_wavefunction_cuda(psi, H, psi_new, grid_size):
    i = cuda.grid(1)
    if i < grid_size:
        psi_new[i] = 0
        for j in range(grid_size):
            psi_new[i] += H[i, j] * psi[j]

class DiracSimulationGPU:
    def __init__(self, mass, grid_size, dt, timesteps):
        self.mass = mass
        self.grid_size = grid_size
        self.dt = dt
        self.timesteps = timesteps
        self.alpha = np.array([[0, 1], [1, 0]])
        self.beta = np.array([[1, 0], [0, -1]])
        self.x = np.linspace(-5, 5, self.grid_size)
        self.psi_t = []
        self.H = None

    def initialize_wavefunction(self, wavefunction_type="gaussian"):
        if wavefunction_type == "gaussian":
            psi_up = np.exp(-self.x**2)
        elif wavefunction_type == "plane_wave":
            psi_up = np.sin(self.x)
        psi_down = np.zeros_like(self.x)
        self.psi = np.vstack([psi_up, psi_down])
        self.psi_t.append(self.psi)

    def build_hamiltonian(self, potential_type="harmonic"):
        dx = self.x[1] - self.x[0]
        T = -1j * np.kron(self.alpha, np.eye(self.grid_size)) / (2 * dx)

        if potential_type == "harmonic":
            V = 0.5 * self.mass * np.kron(np.eye(2), self.x**2)
        elif potential_type == "step":
            V = np.where(self.x > 0, 1, 0) * np.kron(self.beta, np.eye(self.grid_size))
        
        self.H = T + V

    def evolve(self):
        psi_gpu = cuda.to_device(self.psi)
        H_gpu = cuda.to_device(self.H)
        psi_new_gpu = cuda.device_array(self.psi.shape)
        
        threads_per_block = 32
        blocks_per_grid = (self.grid_size + (threads_per_block - 1)) // threads_per_block

        for t in range(self.timesteps):
            evolve_wavefunction_cuda[blocks_per_grid, threads_per_block](psi_gpu, H_gpu, psi_new_gpu, self.grid_size)
            psi_gpu = psi_new_gpu

        psi_final = psi_gpu.copy_to_host()
        return psi_final, self.x
