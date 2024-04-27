import numpy as np
import matplotlib.pyplot as plt

def main():
    """ Superconductor simulation using the spectral method """
    
    # Simulation parameters
    N = 400         # Spatial resolution
    t = 0           # Current time of the simulation
    tEnd = 100     # Time at which simulation ends
    dt = 0.1        # Timestep
    tOut = 0.2      # Draw frequency
    alpha = 0.1     # Superconductor param 1
    beta = 1.5      # Superconductor param 2
    plotRealTime = True  # Switch on for plotting as the simulation goes along
    np.random.seed(917)
    
    # Domain [0,200] x [0,200]
    L = 200
    xlin = np.linspace(0, L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                   # Chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)
    
    # Initial Condition
    psi = 1e-2 * np.random.randn(N, N)
    
    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    
    # Number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
    # Prep figure
    fig = plt.figure(figsize=(4, 4), dpi=150)
    outputCount = 1
    # Simulation Main Loop
    for i in range(Nt):
        # Transform to Fourier space
        psi_k = np.fft.fftn(psi)
        
        # Linear step
        psi_k = np.exp(-1j * dt * (kSq * (alpha - 1j) + 1j)) * psi_k
        
        # Transform back to physical space
        psi = np.fft.ifftn(psi_k)
        
        # Nonlinear step
        psi = psi - dt * (1 - 1j * beta) * np.abs(psi)**2 * psi
        
        # Update time
        t += dt
        
        # Plot in real time
        
        if plotRealTime and (t >= outputCount * tOut or i == Nt - 1):
            plt.cla()
            plt.imshow(np.abs(psi), cmap='bwr', extent=(0, L, 0, L))
            #plt.colorbar()
            plt.clim(0, 1)
            plt.pause(0.004)
            outputCount += 1
    
    # Save figure
    plt.savefig('superconductorSpectral.png', dpi=240)
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()