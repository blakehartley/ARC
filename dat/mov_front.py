import sys
import numpy as np
import matplotlib.pyplot as plt
#import yt
#from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper

fold = "./hydrogen/"

def interp(X, Y, i, x):
	return Y[i] + (Y[i+1]-Y[i])*(x-X[i])/(X[i+1]-X[i])

x_array = []
for arg in range(int(sys.argv[1]),int(sys.argv[2])):
	arr = np.fromfile("./hydrogen/"+"xhi%03d.bin" % arg, dtype=np.float32)
	temp = np.fromfile("./photons/"+"temp%03d.bin" % arg, dtype=np.float32)
	
	DIMX = int((arr.size+1)**(1.0/3.0))
	DIMX2 = int(DIMX/2)
	R = np.linspace(0.0, 6.6, DIMX)
	arr = 1.0 - arr.reshape(DIMX, DIMX, DIMX)
	temp = temp.reshape(DIMX, DIMX, DIMX)
	xHII = arr[0,0,:]
	#xHII = arr[DIMX2,DIMX2,DIMX2:]
	x_array.append(xHII)

x_array = np.array(x_array)
np.savetxt("./profiles/run_0.01_bdf.txt", x_array)
sys.exit()
#rr = 3300*np.arange(0,DIMX2)/DIMX2
rr = 6600*np.arange(0,DIMX)/DIMX

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [])
#plt.loglog()
plt.semilogy()

def init():
	ax.set_xlim(0, 6600)
	#ax.set_ylim(0, 1.1)
	ax.set_ylim(1.e-4,1.e0)
	ax.set_ylabel("$x_{HI}$")
	ax.set_xlabel("pc")
	return ln,

def update(frame):
	#xdata.append(frame)
	#ydata.append(np.sin(frame))
	xdata = rr
	ydata = 1.0 - x_array[frame]
	ln.set_data(xdata, ydata)
	return ln,

ani = FuncAnimation(fig, update, frames=np.array(int(sys.argv[2])-int(sys.argv[1])),
					init_func=init, blit=True)
ani.save("vis/anim.gif", fps=2.0)
plt.show()
