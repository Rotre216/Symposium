import numpy as np
import matplotlib.pyplot as plt

def exit_all(event):
    """ exits the program """
    raise SystemExit

def superconductor_simulation(alpha, beta):
    """ Superconductor simulation """
    
    # Simulation parameters
    N = 400            # Spatial resolution
    t = 0              # current time of the simulation
    tEnd = 100         # time at which simulation ends
    dt = 0.2           # timestep
    tOut = 0.2         # draw frequency
    plotRealTime = True  # switch on for plotting as the simulation goes along
    np.random.seed(917)
    # Domain [0,200] x [0,200]
    L = 200
    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    
    # Initial Condition
    psi = 1e-2 * np.random.randn(N, N)
    V = -(1.j + beta) * np.abs(psi) ** 2  # non-linear part of the equation
    
    # Variables in Fourier space
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx ** 2 + ky ** 2
    
    # Number of steps
    Nt = int(np.ceil(tEnd / dt))
    
    fig = plt.figure(figsize=(4, 4), dpi=150)
    #fig.canvas.mpl_connect('close_event', exit_all)
    outputCount = 1
    
    for i in range(Nt):  # Split-step method
        
        # Total calculation for a half step
        psi = np.exp(-1.j * dt / 2.0 * V) * psi
        
        # Linear portion calculation in Fourier space for a step
        psihat = np.fft.fftn(psi)
        psihat = np.exp(dt * (-1.j * (kSq * (alpha - 1.j) + 1.j))) * psihat
        psi = np.fft.ifftn(psihat)
        
        # Non-linear part calculation
        V = -(1.j + beta) * np.abs(psi) ** 2
        
        # Total equation calculation for a half step
        psi = np.exp(-1.j * dt / 2.0 * V) * psi
        
        # update time
        t += dt
        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(np.abs(psi), cmap='bwr', extent=(0, L, 0, L))
            plt.clim(0, 1)
            ax = plt.gca()
            ax.get_xaxis().set_visible(True)
            ax.get_yaxis().set_visible(True)
            ax.set_aspect('equal')
            plt.pause(0.001)
            outputCount += 1
    
    # Save the final figure
    plt.savefig(f'a0.1b1.5.png', dpi=240)
    plt.close()  # Close the figure to prevent it from displaying

def main():
	beta = 1.5
	alpha = 0.1
	superconductor_simulation(alpha, beta)

if __name__ == "__main__":
    main()
