import sys
import numpy as np
import matplotlib.pyplot as plt
#import yt
#from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
#from yt.visualization.volume_rendering.api import PointSource
#from yt.visualization.volume_rendering.api import Scene, BoxSource

fold = "./hydrogen/"
#fold = "./photons/"

#h=0.7
h=1.0
d=10.0
x=d/h

DIMX = 256

slice0 = int(sys.argv[1])

for arg in range(int(sys.argv[1]), (int(sys.argv[2]))):

	
	slist = np.genfromtxt("./photons/slist%02d.dat" % arg)
	slist = np.float_(slist)
	points = slist[:,0:3]*x/DIMX
	print(slist)

	xx = slist[:, 0]
	yy = slist[:, 1]
	zz = slist[:, 2]

	# Creating figure
	fig = plt.figure(figsize = (10, 10))
	ax = plt.axes(projection ="3d")
	
	# Creating plot
	ax.scatter3D(xx, yy, zz, color = "black")
	#plt.title("simple 3D scatter plot")

	plt.savefig("./vis/scatter%03d" % (arg-slice0))
	plt.close()