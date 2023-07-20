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

DIMX = 256

tracking_array = np.zeros((4, DIMX, DIMX))
tracking_array[:,:] = 31
slice_num = 50

for arg in range(int(sys.argv[1]), (int(sys.argv[1]))+1):
	if(arg % 1) != 0:
		continue
	fold = "../hydrogen/"
	arr_h = np.fromfile(fold+"xhi%03d.bin" % arg, dtype=np.double)
	DIMX = int((arr_h.size+1)**(1.0/3.0))
	print("<x_hii> = {}".format(np.sum(arr_h)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_h), np.max(arr_h)))
	#arr[arr<1.e-6] = 1.e-6
	arr_h = arr_h.reshape(DIMX,DIMX,DIMX)
	arr_h = np.swapaxes(arr_h, 0, 2)

	fold = "../helium/"
	arr_he = np.fromfile(fold+"xhei%03d.bin" % arg, dtype=np.double)
	arr_he = arr_he
	print("<x_heii> = {}".format(np.sum(arr_he)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_he), np.max(arr_he)))
	arr_he = arr_he.reshape(DIMX,DIMX,DIMX)
	arr_he = np.swapaxes(arr_he, 0, 2)

	'''fold = "../photons/"
	arr_energy = np.fromfile(fold+"temp%03d.bin" % arg, dtype=np.float32)
	arr_energy = arr_energy
	print("<energy> = {}".format(np.sum(arr_energy)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_energy), np.max(arr_energy)))
	arr_energy = arr_energy.reshape(DIMX,DIMX,DIMX)
	arr_energy = np.swapaxes(arr_energy, 0, 2)'''
	DIMX = 256
	fold = "../../../nanoJubilee/GridDensities/256/"
	arr_n = np.fromfile(fold + "GridDensities_256_%03d.bin" % int(arg/10), dtype=np.float32)
	print("<n> = {}".format(np.sum(arr_n)/(DIMX)**3))
	arr_n = arr_n.reshape(DIMX,DIMX,DIMX)
	arr_n = np.swapaxes(arr_n, 0, 2)
	dens_slice=arr_n[:,:,slice_num]

	print("<n*xhii> = {}".format(np.sum(arr_n*arr_h)/(DIMX)**3))

	#for i in range(0,256):
	#	print("{} {}".format(i, np.mean(arr_h[:,:,i])))
	
	#plt.imshow(slice, cmap='hot', interpolation='nearest')
	y, x = np.meshgrid(np.linspace(0, 10, DIMX), np.linspace(0, 10, DIMX))
	#y, x = np.arange(0,DIMX), np.arange(0,DIMX)

	fig, ax1 = plt.subplots(1,1, figsize=(8,8))
	#z_min, z_max = np.abs(dens_slice).min(), np.abs(dens_slice).max()
	#c = ax1.pcolormesh(x, y, dens_slice, cmap='RdBu', vmin=z_min, vmax=z_max)
	#fig.colorbar(c, ax=ax1, label='Overdensity')
	'''
	fig, ax2, = plt.subplots(1,1, figsize=(8,8))'''
	ne_density = arr_h*(0.75*arr_h+0.25*0.25*arr_he)*arr_n
	ne_mean = np.mean(ne_density[:,:,:], axis=0)

	tracking_array = ne_mean
	z_min, z_max = np.abs(tracking_array).min(), np.abs(tracking_array).max()
	print("av of {} = {}".format(arg, np.mean(tracking_array)))
	z_min = 0
	z_max = 4
	c = ax1.pcolormesh(x, y, tracking_array, cmap='RdBu', vmin=z_min, vmax=z_max)
	ax1.axis([x.min(), x.max(), y.min(), y.max()])
	fig.colorbar(c, ax=ax1, label='$z$ of $(x_{HII} > 0.9)$')

	#plt.savefig("lya_plot.png")

	#h=0.7
	h=1.0
	d=10.0
	x=d/h
	
	data	= dict(fraction = arr_h, density = ne_density)
	bbox	= np.array([[0.,x],[0.,x],[0.,x]])
	ds	= yt.load_uniform_grid(data, arr_h.shape, length_unit="Mpc", bbox=bbox, nprocs=1)
	
	prj = yt.ProjectionPlot(ds, "z", ["density"])
	#prj.set_cmap("density", "algae")
	prj.set_cmap("density", "RED TEMPERATURE")
	prj.set_zlim("density", zmin=(1.e-50), zmax=4.e-48)
	prj.annotate_grids(cmap=None)
	prj.save('./proj_ne')