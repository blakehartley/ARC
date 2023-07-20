import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.genfromtxt("./profiles/run_0.01.txt")
data1 = np.genfromtxt("./profiles/run_0.01_bdf.txt")
DIMX = data.shape[1]
print(data.shape)

rr = 6600*np.arange(0,DIMX)/DIMX

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], label=r'$dt=0.01\;{\rm Myr}$')
ln1, = plt.plot([], [], label=r'$dt=0.1\;{\rm Myr}$')
#plt.loglog()
plt.semilogy()

def init():
	ax.set_xlim(0, 6600)
	#ax.set_ylim(0, 1.1)
	ax.set_ylim(1.e-4,1.e0)
	ax.set_ylabel("$x_{HI}$")
	ax.set_xlabel("pc")
	ax.legend(loc=4)
	return ln,

def update(frame):
	#xdata.append(frame)
	#ydata.append(np.sin(frame))
	xdata = rr
	ydata = 1.0 - data[frame]
	ln.set_data(xdata, ydata)
	
	ydata = 1.0 - data1[frame]
	ln1.set_data(xdata, ydata)
	return ln,

ani = FuncAnimation(fig, update, frames=data.shape[0],
					init_func=init, blit=True)
ani.save("vis/anim_0.01_bdf.gif", fps=2.0)
#plt.show()
