import sys
import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper

fold = "./hydrogen/"
#fold = "./photons/"

#h=0.7
h=1.0
d=10.0
x=d/h
nstart = int(sys.argv[1])
for arg in range(nstart,int(sys.argv[2])):
	arr = np.fromfile(fold+"xhi%03d.bin" % arg, dtype=np.float32)
	DIMX = int((arr.size+1)**(1.0/3.0))

	arr[arr<1.e-6] = 1.e-6
	#arr = 1.0 - arr
	arr = arr.reshape(DIMX,DIMX,DIMX)

	#for zslice in (0.01, 0.16, 0.49, 0.51, 0.99):
	#for zslice in (0.01, 0.15, 0.49, 0.51, 0.99):
	for zslice in (0.01, 0.15, 0.49, 0.51, 0.99):
		data	= dict(fraction = arr)
		bbox	= np.array([[0.,x],[0.,x],[0.,x]])
		ds	= yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=1)
		
		slc	= yt.SlicePlot(ds, "x", ["fraction"], center=[x/2, x/2, x*zslice])
		slc.set_cmap("fraction", "algae")
		slc.set_zlim('fraction',1.e-4,1.e0)
		#slc.annotate_grids(cmap=None)
		slc.save('./vis/slice_{0}_{1:02d}'.format(zslice,arg-nstart))
	
		'''
		slc	= yt.SlicePlot(ds, "y", ["fraction"], center=[x/2, x/2, x/2])
		slc.set_cmap("fraction", "algae")
		#slc.annotate_grids(cmap=None)
		slc.save('./vis/'+str(arg)+'y')
		'''
		'''
		norm = [-1.0, 0.0+1.e-10, 0.0]
		north = [0,1,0]
		slc	= yt.SlicePlot(ds, norm, ["fraction"], center=[x/2, x/2, x/2],
							width=(1000*x, 'kpc'), north_vector=north)
		slc.set_cmap("fraction", "algae")
		#slc.annotate_grids(cmap=None)
		slc.save('./vis/'+str(arg))
		'''
	
	prj = yt.ProjectionPlot(ds, "x", ["fraction"])
	prj.set_cmap("fraction", "algae")
	prj.annotate_grids(cmap=None)
	prj.save('./vis/proj_{:02d}x'.format(arg-nstart))
'''
// Returns redshift as a function of time in Myr
double red(double time){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	time = time*yrtos*1.e6;
	double redshift = pow((3.*h0*pow(omegam,0.5)*time/2.),-2./3.)-1.;
	return redshift;
}

// Returns time in Myr as a function of redshift
double tim(double redshift){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	double time = 2.*pow((1.+redshift),-3./2.)/(3.*h0*pow(omegam,0.5));
	time = time/(yrtos*1.e6);
	return time;
}'''
