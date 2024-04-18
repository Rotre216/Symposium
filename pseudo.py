import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own Superconductor Simulation (With Python)
Philip Mocz (2023), @PMocz555


Simulate a simplified version of the 
time-dependent complex Ginzburg-Landau equation 
with the Pseudo Spectral method

d psi / d t = (1+i*alpha) * nabla^2 psi + psi - (1-i*beta)*|psi|^2*psi
"""

def exit_all(event):
    """ exits the program """
    raise SystemExit

def initialize_simulation_parameters():
    """Initializes parameters for the superconductor simulation."""
    parameters = {
        'N': 400,            # Spatial resolution
        'tEnd': 100,         # End time of the simulation
        'dt': 0.2,           # Time step
        'tOut': 0.2,         # Output frequency
        'alpha': 0.1,        # Superconductor parameter 1
        'beta': 1.5,         # Superconductor parameter 2
        'plotRealTime': True,# Enable real-time plotting
        'random_seed': 917   # Seed for random number generation
    }
    return parameters

def initialize_domain(N):
    """Initializes the spatial domain."""
    L = 200
    x = np.linspace(0, L, num=N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    return xx, yy

def initialize_initial_conditions(N, random_seed, beta):
    """Initializes the initial condition for the superconductor."""
    np.random.seed(random_seed)
    psi = 1e-2 * np.random.randn(N, N)
    V = -(1j + beta) * np.abs(psi) ** 2
    return psi, V

def initialize_fourier_space_variables(N, L):
    """Initializes variables in Fourier space."""
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx ** 2 + ky ** 2
    return kx, ky, kSq

def main():
    """Main function to run the superconductor simulation."""
    parameters = initialize_simulation_parameters()
    Nt = int(np.ceil(parameters['tEnd'] / parameters['dt']))

    xx, yy = initialize_domain(parameters['N'])
    psi, V = initialize_initial_conditions(parameters['N'], parameters['random_seed'], parameters['beta'])
    kx, ky, kSq = initialize_fourier_space_variables(parameters['N'], 200)

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.mpl_connect('close_event', exit_program)
    output_count = 1

    for i in range(Nt):
        psi = np.exp(-1j * parameters['dt'] / 2.0 * V) * psi
        psihat = np.fft.fft2(psi)
        drift_phase = parameters['dt'] * (-1j * (kSq * (parameters['alpha'] - 1j) + 1j))
        psihat *= np.exp(drift_phase)
        psi = np.fft.ifft2(psihat).real
        V = -(1j + parameters['beta']) * np.abs(psi) ** 2
        psi = np.exp(-1j * parameters['dt'] / 2.0 * V) * psi
        t = (i + 1) * parameters['dt']

        if parameters['plotRealTime'] and t >= output_count * parameters['tOut']:
            plt.clf()
            plt.imshow(np.abs(psi), cmap='bwr', extent=(0, 200, 0, 200))
            plt.clim(0, 1)
            plt.gca().invert_yaxis()
            plt.gca().set_xlabel('x')
            plt.gca().set_ylabel('y')
            plt.title(f'Time: {t}')
            plt.colorbar(label='|psi|')
            plt.pause(0.001)
            output_count += 1

    plt.savefig('superconductor.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
