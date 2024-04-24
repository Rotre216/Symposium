import numpy as np
import matplotlib.pyplot as plt

def exit_all(event):
    """ exits the program """
    plt.close()
    raise SystemExit

def initialize_simulation_parameters():
    """
    Initializes parameters for the superconductor simulation.
    
    Returns:
        parameters (dict): A dictionary containing simulation parameters.
    """
    parameters = {
        'N': 400,            # Spatial resolution: Number of grid points in each spatial dimension
        'tEnd': 100,         # End time of the simulation
        'dt': 0.2,           # Time step
        'tOut': 0.2,         # Output frequency: Time interval for displaying results
        'alpha': 0.1,        # Superconductor parameter 1
        'beta': 1.5,         # Superconductor parameter 2
        'B_ext': 0.05,       # Magnitude of the external magnetic field
        'plotRealTime': True,# Enable real-time plotting
        'random_seed': 917   # Seed for random number generation
    }
    return parameters

def initialize_domain(N):
    """
    Initializes the spatial domain.
    
    Args:
        N (int): Number of grid points in each spatial dimension.
        
    Returns:
        xx, yy (ndarray): Meshgrid arrays representing spatial coordinates.
    """
    L = 200  # Length of the spatial domain
    x = np.linspace(0, L, num=N, endpoint=False)  # Spatial coordinate array
    xx, yy = np.meshgrid(x, x)  # 2D meshgrid arrays representing spatial coordinates
    return xx, yy

def initialize_initial_conditions(N, random_seed, beta):
    """
    Initializes the initial condition for the superconductor.
    
    Args:
        N (int): Number of grid points in each spatial dimension.
        random_seed (int): Seed for random number generation.
        beta (float): Superconductor parameter 2.
        
    Returns:
        psi (ndarray): Initial order parameter.
        V (ndarray): External potential.
    """
    np.random.seed(random_seed)
    psi = 1e-2 * np.random.randn(N, N)  # Random initial order parameter
    V = -(1j + beta) * np.abs(psi) ** 2  # External potential
    return psi, V

def initialize_fourier_space_variables(N, L):
    """
    Initializes variables in Fourier space.
    
    Args:
        N (int): Number of grid points in each spatial dimension.
        L (float): Length of the spatial domain.
        
    Returns:
        kx, ky (ndarray): 2D arrays representing wave vectors.
        kSq (ndarray): Square of the wave vector.
    """
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)  # Linearly spaced wave vector
    kx, ky = np.meshgrid(klin, klin)  # 2D meshgrid arrays representing wave vectors
    kx = np.fft.ifftshift(kx)  # Shift zero frequency component to the center
    ky = np.fft.ifftshift(ky)  # Shift zero frequency component to the center
    kSq = kx ** 2 + ky ** 2  # Square of the wave vector
    return kx, ky, kSq

def main():
    """
    Main function to run the superconductor simulation.
    """
    parameters = initialize_simulation_parameters()
    Nt = int(np.ceil(parameters['tEnd'] / parameters['dt']))

    # Initialize spatial domain
    xx, yy = initialize_domain(parameters['N'])
    # Initialize initial conditions
    psi, V = initialize_initial_conditions(parameters['N'], parameters['random_seed'], parameters['beta'])
    # Initialize variables in Fourier space
    kx, ky, kSq = initialize_fourier_space_variables(parameters['N'], 200)

    # Set up plotting
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.mpl_connect('close_event', exit_all)
    output_count = 1

    # Main simulation loop
    for i in range(Nt):
        # Update psi in Fourier space
        psi = np.exp(-1j * parameters['dt'] / 2.0 * V) * psi
        psihat = np.fft.fft2(psi)
        drift_phase = parameters['dt'] * (-1j * (kSq * (parameters['alpha'] - 1j) + 1j))
        psihat *= np.exp(drift_phase)
        psi = np.fft.ifft2(psihat).real
        # Update external potential with an external magnetic field
        V = -(1j + parameters['beta']) * np.abs(psi) ** 2 + parameters['B_ext'] * (xx * ky - yy * kx)
        psi = np.exp(-1j * parameters['dt'] / 2.0 * V) * psi
        t = (i + 1) * parameters['dt']

        # Real-time plotting if enabled
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

    # Save and display final result
    plt.savefig('superconductor.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()