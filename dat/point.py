import sys
import numpy as np
import matplotlib.pyplot as plt

DIMX = 256

slist = np.genfromtxt("./photons/slist%02d.dat" % 20)
slist = np.float_(slist)

indxyz = slist[:,0:3].astype(int)
indarr = DIMX*DIMX*indxyz[:,2] + DIMX*indxyz[:,1] + indxyz[:,0]

ind = indarr[0]

for arg in range(int(sys.argv[1]), (int(sys.argv[2]))):
	xhi = np.fromfile("./hydrogen/xhi%03d.bin" % arg, dtype=np.float32)
	T = np.fromfile("./photons/temp%03d.bin" % arg, dtype=np.float32)
	#flux = np.fromfile("./photons/flux%03d.bin" % arg, dtype=np.float32)
	flux = np.fromfile("./photons/temp%03d.bin" % arg, dtype=np.float32)
	
	for i in range(0,indarr.size):
		print("{0}\t{1}\t{2}\t".format(xhi[indarr[i]], flux[indarr[i]], T[indarr[i]]), end='')
	print('', end="\n")
