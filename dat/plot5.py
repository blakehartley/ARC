import sys
import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import PointSource
from yt.visualization.volume_rendering.api import Scene, VolumeSource

fold = "./hydrogen/"
#fold = "./photons/"

#h=0.7
h=1.0
d=10.0
x=d/h

slice0 = 10
nrot = 5

for arg in range(int(sys.argv[1]), (int(sys.argv[1])+1)):
	arr = np.fromfile(fold+"xhi%03d.bin" % arg, dtype=np.float32)
	DIMX = int((arr.size+1)**(1.0/3.0))
	#arr = np.fromfile(fold+"temp%03d.bin" % arg, dtype=np.float32)
	print(np.sum(1-arr)/(DIMX)**3)
	arr[arr<1.e-6] = 1.e-6
	arr = arr.reshape(DIMX,DIMX,DIMX)
	arr = np.swapaxes(arr, 0, 2)
	
	'''arrD = np.fromfile("./GridDensities/GridDensities_256_%03d.bin" % arg, dtype=np.float32)
	arrD = arrD.reshape(DIMX,DIMX,DIMX)*(3.08567758e24)**3.0
	arrD = np.swapaxes(arrD, 0, 2)'''
	
	'''dim = int(DIMX/2)
	arr = arr[0:dim, 0:dim, 0:dim]
	arrD = arrD[0:dim, 0:dim, 0:dim]'''
	
	bbox = np.array([[0.,x],[0.,x],[0.,x]])

	bounds = (1.e-1,1.e0)
	arr = 1.0 - arr
	#data	= dict(fraction = arr, density = arrD)
	data	= dict(fraction = arr)
	ds	= yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=1)
	
	norm = [-1.0, 0.0+1.e-10, 0.0]
	north = [0,1,0]
	slc	= yt.SlicePlot(ds, norm, ["fraction"], center=[x/2.05, x/2, x/2],
						width=(1000*x, 'kpc'), north_vector=north)
	slc.set_cmap("fraction", "algae")
	#slc.annotate_grids(cmap=None)
	slc.save('./vis/A'+str(arg))
	
	norm = [-1.0, 0.0+1.e-10, 0.0]
	north = [0,1,0]
	slc	= yt.SlicePlot(ds, norm, ["fraction"], center=[x/1.95, x/2, x/2],
						width=(1000*x, 'kpc'), north_vector=north)
	slc.set_cmap("fraction", "algae")
	#slc.annotate_grids(cmap=None)
	slc.save('./vis/B'+str(arg))
