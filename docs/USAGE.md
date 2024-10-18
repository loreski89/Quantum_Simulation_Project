
# Quantum Simulation Project - Usage Guide

## Prerequisites
To run the quantum simulations in this project, ensure you have the following installed:
- Python 3.8 or higher
- `numpy`
- `numba` (with CUDA support)

You also need a compatible GPU to leverage the GPU acceleration features of this project.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/loreski89/Quantum_Simulation_Project.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the project locally using:
   ```bash
   pip install .
   ```

## Running the Simulations
To run the Dirac equation simulation, navigate to the `src/` folder and run:

```bash
python dirac_simulation.py
```

## Modifying the Simulation
You can modify the simulation parameters such as the grid size, time steps, and mass by editing the `dirac_simulation.py` file inside the `src/` folder.

## Contact
If you have any issues or questions, feel free to contact the project author at **lorenzo.schivo@me.com**.
