import sys
import numpy as np
import matplotlib.pyplot as plt
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import PointSource
from yt.visualization.volume_rendering.api import Scene, BoxSource

fold = "./hydrogen/"
#fold = "./photons/"

#h=0.7
h=1.0
d=10.0
x=d/h

slice0 = int(sys.argv[1])
nrot = 5
#indices = {75, 125, 175, 225, 275, 325, 375, 425}
indices = {225}
#for arg in range(int(sys.argv[3])+slice0, (int(sys.argv[2]))):
for arg in indices:
	if(arg % 5) != 0:
		continue
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

	'''fold = "./photons/"
	arr_energy = np.fromfile(fold+"temp%03d.bin" % arg, dtype=np.float32)
	arr_energy = arr_energy
	print("<energy> = {}".format(np.sum(arr_energy)/(DIMX)**3))
	print("min, max = {}, {}".format(np.min(arr_energy), np.max(arr_energy)))
	arr_energy = arr_energy.reshape(DIMX,DIMX,DIMX)
	arr_energy = np.swapaxes(arr_energy, 0, 2)'''
	DIMX = 256
	fold = "../../nanoJubilee/GridDensities/256/"
	arr_n = np.fromfile(fold + "GridDensities_256_%03d.bin" % int(arg/10), dtype=np.float32)
	print("<n> = {}".format(np.sum(arr_n)/(DIMX)**3))
	arr_n = arr_n.reshape(DIMX,DIMX,DIMX)
	arr_n = np.swapaxes(arr_n, 0, 2)
	
	bbox = np.array([[0.,x],[0.,x],[0.,x]])
	arr_ne = arr_n*(0.75*arr_h + 0.25*0.25*arr_he)
	print("min, max = {}, {}".format(np.min(arr_ne), np.max(arr_ne)))

	bounds = (1.e-2,1.e0)
	#arr_h = 1.0 - arr_h
	#data	= dict(fraction = arr, density = arrD)
	data	= dict(fraction = arr_h, density = arr_ne)
	ds	= yt.load_uniform_grid(data, arr_h.shape, length_unit="Mpc", bbox=bbox, nprocs=1)
	print(ds)
	#sc = yt.create_scene(ds, 'fraction', lens_type='perspective')
	sc = yt.create_scene(ds, 'fraction', lens_type='plane-parallel')
	
	tf = yt.ColorTransferFunction(np.log10(bounds))
	#cmap = plt.cm.get_cmap('algae')
	cmap = plt.cm.get_cmap('RdYlBu')
	#cmap = plt.cm.get_cmap('viridis')
	#cmap = plt.cm.get_cmap('PRISM')
	cen = [2.e-2, 5.e-1, 8.e-1]
	col = [cmap(1.0), cmap(0.2), cmap(0.0)]
	h = [1.e0, 0.0e-1, 1.0e-1]
	w = [0.001, 0.001, 0.001]
	
	for i in range(0, len(cen)):
		tf.add_gaussian(np.log10(cen[i]), width=w[i], height=[col[i][0], col[i][1], col[i][2], h[i]])
	def linramp(vals, minval, maxval):
		return (vals - vals.min())/ (vals.max() - vals.min())
		#return (vals.max() - vals)/ (vals.max() - vals.min())
		#return (vals.max() - vals)/ (vals.max() - vals.min())
	
	#tf.map_to_colormap(
	#	np.log10(1.e-2), np.log10(1.e0), colormap=cmap, scale_func=linramp
	#)
	#np.log(float(N)/(i+0.5))
	source = sc[0]
	source.set_log(True)
	source.tfh.tf = tf
	source.tfh.bounds = bounds
	source.tfh.plot('./figures/transfer_function%03d.png' % arg, profile_field='fraction')
	
	#sc.annotate_text([0,0], 'Hello')
	
	slist = np.genfromtxt("./photons/slist%02d.dat" % (int(arg/5)))
	slist = np.float_(slist)
	print(slist.ndim)
	if(slist.ndim == 1):
		slist = np.array([slist, slist])
	
	points = slist[:,0:3]*x/DIMX

	#print(slist)

	'''points = points*2
	points = points[points[:,0]<d]
	points = points[points[:,1]<d]
	points = points[points[:,2]<d]'''
	
	#cam = sc.add_camera(ds, lens_type='perspective')
	cam = sc.camera
	#cam=lens = 'perspective'
	#cam.switch_
	#DIM = 256
	DIM = 1024
	cam.resolution = [DIM,DIM]
	cam.set_lens('perspective')
	#sc.save('./vis/srendering%03d.png' % arg, sigma_clip = 1)

	colors = np.zeros([points[:,0].size, 4])
	colors[:,0:3] = 1.0
	#colors[:,3] = 0.075*slist[:,4]/slist[0][4]+0.025
	colors[:,3][points[:,0]>0] = 0.02
	rad = np.zeros(slist[:,0].size)
	for i in range(0, points[:,0].size):
		rad[i] = int(2+2*np.log(slist[i][4])/np.log(slist[0][4]))
	rad = rad.astype(int)
	
	#verts = PointSource(points, colors = colors, color_stride = 1, radii = int(DIM/512))
	verts = PointSource(points, colors = colors, color_stride = 1, radii = 1)
	#sc.add_source(verts)
	
	cam.set_position((0.51, 0.51, 0.51))
	
	#DTheta = 10.0*np.pi/180.0
	#NTheta = 10
	DTheta = 0*2.0*np.pi/180.0
	NTheta = 1
	#cam.rotate((arg-slice0)*DTheta)
	
	j=0
	for i in cam.iter_rotate(DTheta, NTheta):
		frame = NTheta*(arg-slice0) + j
		
		theta = frame*DTheta/NTheta
		dx = np.cos(theta)
		dy = np.sin(theta)
		north_vector = (-dx, -dy, 0.7071)
		orig = np.array([0.5, 0.5, 0.5])		
		vec = np.array([0.7071*dx, 0.7071*dy, 0.5*(1.0 - frame/400.0)])
		cam.set_position(orig + 1.5*vec, north_vector = north_vector)
		#cam.set_focus((0.5, 0.5, 0.5))
		#cam.switch_orientation(normal_vector = normal_vector, north_vector=north_vector)
		
		print(cam)
	
		'''xcam = 5+5*np.cos(theta)
		ycam = 5+5*np.sin(theta)
		cam.set_position(xcam, ycam, 10.0)
		cam.set_focus(5, 5, 5)
		cam.switch_orientation()'''
	
		#sc.camera.width = (10, 'Mpc')
		#sc.camera.switch_orientation()
	
		#sc.annotate_grids(ds, alpha=0.01)
		sc.render()
		sc.save('./figures/rendering%03d.png' % frame, sigma_clip=1)
		j+=1
		#sc.save('./rendering%03d_y2.png' % arg, sigma_clip=2)
		#sc.save('./rendering%03d_3.png' % arg, sigma_clip=3)
		#sys.exit()
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
