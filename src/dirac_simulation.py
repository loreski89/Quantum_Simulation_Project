
import numpy as np

class DiracSimulationCPU:
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
        print("Inizializzazione completata")

    def initialize_wavefunction(self, wavefunction_type="gaussian"):
        print("Inizializzazione della funzione d'onda")
        if wavefunction_type == "gaussian":
            psi_up = np.exp(-self.x**2)
        elif wavefunction_type == "plane_wave":
            psi_up = np.sin(self.x)
        psi_down = np.zeros_like(self.x)
        self.psi = np.vstack([psi_up, psi_down])  # psi sarà ora di dimensione (2, grid_size)
        self.psi_t.append(self.psi)
        print("Funzione d'onda inizializzata")

    def build_hamiltonian(self, potential_type="harmonic"):
        print("Costruzione dell'Hamiltoniano")
        dx = self.x[1] - self.x[0]
        T = -1j * np.kron(self.alpha, np.eye(self.grid_size)) / (2 * dx)

        # V deve avere la stessa dimensione di T, quindi (200, 200)
        if potential_type == "harmonic":
            V = 0.5 * self.mass * np.diag(self.x**2)  # Matrice diagonale di V (100, 100)
            V = np.kron(np.eye(2), V)  # Ora V sarà (200, 200)
        elif potential_type == "step":
            V = np.where(self.x > 0, 1, 0)
            V = np.kron(self.beta, np.diag(V))  # Ora V sarà (200, 200)

        self.H = T + V
        print("Hamiltoniano costruito")

    def evolve(self):
        print("Inizio evoluzione temporale")
        psi_new = np.zeros_like(self.psi)

        for t in range(self.timesteps):
            print(f"Passo temporale {t+1}/{self.timesteps}")
            psi_new = np.dot(self.H, self.psi.flatten()).reshape(2, self.grid_size)  # Allinea correttamente le dimensioni
            self.psi = psi_new

        print("Evoluzione completata")
        return self.psi, self.x

if __name__ == "__main__":
    print("Simulazione avviata...")
    sim = DiracSimulationCPU(1.0, 100, 0.01, 10)
    sim.initialize_wavefunction()
    print("Funzione d'onda pronta...")
    sim.build_hamiltonian()
    print("Hamiltoniano pronto...")
    result, x_values = sim.evolve()
    print("Simulazione completata!")
