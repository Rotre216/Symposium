import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own Superconductor Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate a simplified version of the 
time-dependent complex Ginzburg-Landau equation 
with the Spectral method

d psi / d t = (1+i*alpha) * nabla^2 psi + psi - (1-i*beta)*|psi|^2*psi
"""

		
def exit_all(event):
	""" exits the program """
	raise SystemExit

def main():
	""" Superconductor simulation """
	
	# Simulation parameters
	N         = 400     # Spatial resolution
	t         = 0       # current time of the simulation
	tEnd      = 100     # time at which simulation ends
	dt        = 0.2     # timestep
	tOut      = 0.2     # draw frequency
	alpha     = 0.1     # superconductor param 1
	beta      = 1.5     # superconductor param 2
	plotRealTime = True # switch on for plotting as the simulation goes along
	np.random.seed(917)
	
	# Domain [0,200] x [0,200]
	L = 200
	xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
	xlin = xlin[0:N]                  # chop off periodic point
	xx, yy = np.meshgrid(xlin, xlin)
	
	# Intial Condition
	psi = 1e-2 * np.random.randn(N,N)
	V = -(1.j+beta)*np.abs(psi)**2 #partie non linéaire de l'équation
	
	#variables dans l'esace de fourier
	klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
	kx, ky = np.meshgrid(klin, klin)
	kx = np.fft.ifftshift(kx)
	ky = np.fft.ifftshift(ky)
	kSq = kx**2 + ky**2
	
	#nombre de pas
	Nt = int(np.ceil(tEnd/dt))
	

	fig = plt.figure(figsize=(4,4), dpi=150)
	fig.canvas.mpl_connect('close_event', exit_all)
	outputCount = 1

	for i in range(Nt): #Split-step method
		
		#première calculation totale pour un demi pas
		psi = np.exp(-1.j*dt/2.0*V) * psi
		
		# calculation de la portion linéaire dans l'espace de fourier pour un pas
		psihat = np.fft.fftn(psi)
		psihat = np.exp(dt * (-1.j* (kSq*(alpha-1.j) + 1.j))) * psihat 
		psi = np.fft.ifftn(psihat)
		
		# calculation de la partie non-linéaire
		V = -(1.j+beta)*np.abs(psi)**2
		
		# calculation de l'équation totale pour un demi pas
		psi = np.exp(-1.j*dt/2.0*V) * psi
		
		# update time
		t += dt
		print(t)
		
		# plot in real time
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			plotThisTurn = True
		if (plotRealTime and plotThisTurn) or (i == Nt-1):
			plt.cla()
			plt.imshow(np.abs(psi), cmap = 'bwr')
			plt.clim(0,1)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.001)
			outputCount += 1
					
	# Save figure
	plt.savefig('superconductor.png',dpi=240)
	plt.show()
	
	return 0
	


if __name__== "__main__":
  main()
