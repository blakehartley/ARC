import sys
import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import PointSource
from yt.visualization.volume_rendering.api import Scene, BoxSource

def red(time):
	h=0.6711
	h0=h*3.246753e-18
	omegam=0.3
	yrtos=3.15569e7
	time = time*yrtos*1.e6
	redshift = pow((3.*h0*pow(omegam,0.5)*time/2.),-2./3.)-1.
	return redshift

def tim(redshift):
	h=0.6711
	h0=h*3.246753e-18
	omegam=0.3
	yrtos=3.15569e7
	time = 2.*pow((1.+redshift),-3./2.)/(3.*h0*pow(omegam,0.5))
	time = time/(yrtos*1.e6)
	return time

def sim_redshift(i):
	x = tim(30) + 2*i
	return(red(x))

#h=0.7
h=1.0
d=10.0
x=d/h

myr_per_output = 2.0

print("Time\t\tS0\t\t\td(x_HII)\tGam_H*dt\tAl_H*dt\t\tRatio\t\td(x_HeII)\tGam_He*dt\tAl_He*dt\tRatio\t\tS0_tot\t\tx_HII\t\tGam_H*T\t\tAlp_H*T\t\tx_HeII\t\tGam_He*T\tAlp_He*T\tFlux/S0\t\tescaped")


for arg in range(int(sys.argv[1]), (int(sys.argv[2]))):
	fold = "./hydrogen/"
	arr_h = np.fromfile(fold+"xhi%03d.bin" % arg, dtype=np.double)
	DIMX = int((arr_h.size+1)**(1.0/3.0))
	print("<x_hii> = {}".format(np.sum(arr_h)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_h), np.max(arr_h)))
	#arr[arr<1.e-6] = 1.e-6
	arr_h = arr_h.reshape(DIMX,DIMX,DIMX)
	arr_h = np.swapaxes(arr_h, 0, 2)

	fold = "./helium/"
	arr_he = np.fromfile(fold+"xhei%03d.bin" % arg, dtype=np.double)
	arr_he = arr_he
	print("<x_heii> = {}".format(np.sum(arr_he)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_he), np.max(arr_he)))
	arr_he = arr_he.reshape(DIMX,DIMX,DIMX)
	arr_he = np.swapaxes(arr_he, 0, 2)

	fold = "./photons/"
	arr_energy = np.fromfile(fold+"temp%03d.bin" % arg, dtype=np.float32)
	arr_energy = arr_energy
	print("<energy> = {}".format(np.sum(arr_energy)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_energy), np.max(arr_energy)))
	arr_energy = arr_energy.reshape(DIMX,DIMX,DIMX)
	arr_energy = np.swapaxes(arr_energy, 0, 2)
	DIMX = 256
	fold = "../../nanoJubilee/GridDensities/256/"
	arr_n = np.fromfile(fold + "GridDensities_256_%03d.bin" % int(myr_per_output*arg/10), dtype=np.float32)
	print("<n> = {}".format(np.sum(arr_n)/(DIMX)**3))
	arr_n = arr_n.reshape(DIMX,DIMX,DIMX)
	arr_n = np.swapaxes(arr_n, 0, 2)

	t = myr_per_output*arg

	print("{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\n"
       .format( t, # Time\t\t
	       		0, # S0\t\t\t
				0, # d(x_HII)\t
				0, # Gam_H*dt\t
				0, # Al_H*dt\t\t
				0, # Ratio\t\t
				0, # d(x_HeII)\t
				0, # Gam_He*dt\t
				0, # Al_He*dt\t
				0, # Ratio\t\t
				0, # S0_tot\t\t
				0, # x_HII\t\t
				0, # Gam_H*T\t\t
				0, # Alp_H*T\t\t
				0, # x_HeII\t\t
				0, # Gam_He*T\t
				0, # Alp_He*T\t
				0, # Flux/S0\t\t
				0)) # escaped"
