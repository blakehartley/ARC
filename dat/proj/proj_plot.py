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

tracking_array = np.load("proj.npy")

i=0
#plt.imshow(slice, cmap='hot', interpolation='nearest')
y, x = np.meshgrid(np.linspace(0, 10, DIMX), np.linspace(0, 10, DIMX))
#y, x = np.arange(0,DIMX), np.arange(0,DIMX)

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
#z_min, z_max = np.abs(dens_slice).min(), np.abs(dens_slice).max()
#c = ax1.pcolormesh(x, y, dens_slice, cmap='RdBu', vmin=z_min, vmax=z_max)
#fig.colorbar(c, ax=ax1, label='Overdensity')
'''
fig, ax2, = plt.subplots(1,1, figsize=(8,8))'''

z_min, z_max = np.abs(tracking_array[i]).min(), np.abs(tracking_array[i]).max()
c = ax1.pcolormesh(x, y, tracking_array[i], cmap='RdBu', vmin=z_min, vmax=z_max)
ax1.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax1, label='$z$ of ($x_{HII}$ > $0.9$')

plt.savefig("proj_plot_{}.png".format(i))

thresholds = [0.9, 0.1, 0.5, 0.9]

for i in range(1,4):
	#plt.imshow(slice, cmap='hot', interpolation='nearest')
	y, x = np.meshgrid(np.linspace(0, 10, DIMX), np.linspace(0, 10, DIMX))
	#y, x = np.arange(0,DIMX), np.arange(0,DIMX)

	fig, ax1 = plt.subplots(1,1, figsize=(8,8))
	#z_min, z_max = np.abs(dens_slice).min(), np.abs(dens_slice).max()
	#c = ax1.pcolormesh(x, y, dens_slice, cmap='RdBu', vmin=z_min, vmax=z_max)
	#fig.colorbar(c, ax=ax1, label='Overdensity')
	'''
	fig, ax2, = plt.subplots(1,1, figsize=(8,8))'''

	z_min, z_max = np.abs(tracking_array[i]).min(), np.abs(tracking_array[i]).max()
	c = ax1.pcolormesh(x, y, tracking_array[i], cmap='RdBu', vmin=z_min, vmax=z_max)
	ax1.axis([x.min(), x.max(), y.min(), y.max()])
	fig.colorbar(c, ax=ax1, label='$z$ of ($<x_{HII}>$ > ' + '{}'.format(thresholds[i]))

	plt.savefig("proj_plot_{}.png".format(i))

"""fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
z_min, z_max = np.abs(dens_slice).min(), np.abs(dens_slice).max()
c = ax1.pcolormesh(x, y, dens_slice, cmap='RdBu', vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax1, label='Overdensity')
'''
fig, ax2, = plt.subplots(1,1, figsize=(8,8))'''

z_min, z_max = np.abs(tracking_array).min(), np.abs(tracking_array).max()
print("av of {} = {}".format(i, np.mean(tracking_array)))
c = ax2.pcolormesh(x, y, tracking_array, cmap='RdBu', vmin=z_min, vmax=z_max)
ax2.axis([x.min(), x.max(), y.min(), y.max()])'''
fig.colorbar(c, ax=ax2, label='$z$ of $(x_{HII} > 0.9)$')""""""

if(False):
	DIMX = 256
	'''bbox = np.array([[0.,x],[0.,x],[0.,x]])

	bounds = (8.e-2,1.e0)
	#arr = 1.0 - arr
	#data	= dict(fraction = arr, density = arrD)
	data	= dict(	x_h = arr_h,
	     			x_he = arr_he,
					energy = arr_energy,
					density = arr_n
	     )
	ds	= yt.load_uniform_grid(data, arr_h.shape, length_unit="Mpc", bbox=bbox, nprocs=1)
	print(ds)
	print(ds.field_list)
	
	plot = yt.PhasePlot(
		ds,
		('stream', 'x_h'),
		('stream', 'density'),
		('stream', 'x_he'),
		weight_field=None,
	)

	plot.save("phase.png")'''
	"""
	#sc = yt.create_scene(ds, 'fraction', lens_type='perspective')
if(False):
	sc = yt.create_scene(ds, 'fraction', lens_type='plane-parallel')
	
	tf = yt.ColorTransferFunction(np.log10(bounds))
	#cmap = plt.cm.get_cmap('algae')
	#cmap = plt.cm.get_cmap('RdYlBu')
	#cmap = plt.cm.get_cmap('viridis')
	#cmap = plt.cm.get_cmap('PRISM')
	'''
	cmap = plt.cm.get_cmap('bds_highcontrast')
	cen = [1.e-1, 8.e-1, 1.e0]
	col = [cmap(0.2), cmap(0.45), cmap(1.0)]
	h = [10.e0, 8.e-1, 3.e-1]
	w = [0.001, 0.001, 0.001]
	'''
	cmap = plt.cm.get_cmap('RdBu')
	cen = [1.e-1, 7.5e-1, 1.e0]
	col = [cmap(1.0), cmap(0.2), cmap(0.0)]
	h = [1.e1, 4.0e-1, 2.0e-1]
	w = [0.001, 0.001, 0.001]
	for i in range(0, len(cen)):
		tf.add_gaussian(np.log10(cen[i]), width=w[i], height=[col[i][0], col[i][1], col[i][2], h[i]])
	#np.log(float(N)/(i+0.5))
	source = sc[0]
	source.tfh.tf = tf
	source.tfh.bounds = bounds
	#source.tfh.plot('./transfer_function%03d.png' % arg, profile_field='fraction')
	
	#sc.annotate_text([0,0], 'Hello')
	
	slist = np.genfromtxt("./photons/slist%02d.dat" % (arg/5))
	slist = np.float_(slist)
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
	sc.add_source(verts)
	
	cam.set_position((0.51, 0.51, 0.51))
	
	#DTheta = 10.0*np.pi/180.0
	#NTheta = 10
	DTheta = 2.0*np.pi/180.0
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
		sc.save('./vis/rendering%03d.png' % frame, sigma_clip=1)
		j+=1
		#sc.save('./rendering%03d_y2.png' % arg, sigma_clip=2)
		#sc.save('./rendering%03d_3.png' % arg, sigma_clip=3)
		#sys.exit()"""
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
