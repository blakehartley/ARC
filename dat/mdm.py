import sys
import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import PointSource
from yt.visualization.volume_rendering.api import Scene, VolumeSource

fold = "./photons/"

for arg in range(int(sys.argv[1]), (int(sys.argv[2]))):
	arr = np.genfromtxt(fold+"slist%02d.dat" % arg)
	
	if(arr.size == 0):
		print(5.e7)
	elif(arr.size == 5):
		mass = arr[4]
		print(arr[4])
	else:
		mass = arr[:,4]
		print(np.min(arr[:,4]))
	

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
