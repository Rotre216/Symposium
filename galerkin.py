import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def exit_all(event):
    """ exits the program """
    raise SystemExit

def superconductor_simulation(alpha, beta):
    """ Superconductor simulation """
    
    # Simulation parameters
    N = 64             # Spatial resolution
    t = 0              # current time of the simulation
    tEnd = 100         # time at which simulation ends
    dt = 0.2           # timestep
    tOut = 0.2         # draw frequency
    plotRealTime = True  # switch on for plotting as the simulation goes along
    np.random.seed(917)
    T = 273
    
    # Domain [0,200] x [0,200]
    L = 200
    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    
    # Chebyshev nodes
    x = np.cos(np.pi * np.arange(N) / N)
    
    # Differentiation matrix
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (x[i] / (x[i] - x[j])) * ((-1) ** (i + j))
    np.fill_diagonal(D, 0.5)
    
    # Mass matrix
    M = np.eye(N)
    M[0, 0] = 2.0
    M[-1, -1] = 2.0
    
    # Initial Condition
    psi = 1e-2 * np.random.randn(N)
    V = -(1.j + beta) * np.abs(psi) ** 2  # non-linear part of the equation
    
    # Number of steps
    Nt = int(np.ceil(tEnd / dt))
    
    fig = plt.figure(figsize=(4, 4), dpi=150)
    #fig.canvas.mpl_connect('close_event', exit_all)
    outputCount = 1
    
    for i in range(Nt):  # Galerkin method
        
        # Galerkin projection
        rhs = (alpha - 1.j) * D @ D @ psi + 1.j * beta * psi ** 3
        
        # Solve the resulting system
        psi = solve(M - dt * rhs, psi)
        
        # update time
        t += dt
        
        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.plot(xlin, np.abs(psi))
            plt.xlabel('x')
            plt.ylabel('|Psi|')
            plt.title(f'Time t = {t:.2f}')
            plt.pause(0.001)
            outputCount += 1
    
    # Save the final figure
    plt.savefig(f'superconductor_simulation_temp.png', dpi=240)
    plt.close()  # Close the figure to prevent it from displaying

def main():
    beta = 3
    alpha = 100
    superconductor_simulation(alpha, beta)

if __name__ == "__main__":
    main()
