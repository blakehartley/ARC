import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./profiles/test/no_correction.dat", skip_header=1).T
label = 'single source'
data1 = np.genfromtxt("./conservation.dat", skip_header=1).T
label1 = 'test'
'''data = np.genfromtxt("./profiles/euler/run_0.01_const.txt", skip_header=1).T
label = 'dt = 0.01 Myr'
data1 = np.genfromtxt("./profiles/euler/run_0.01_uncorrected.txt", skip_header=1).T
label1 = 'dt = 0.01 Myr (uncorrected)'
#data1 = np.genfromtxt("./profiles/run_0.1_sub_0.5_1.0.txt", skip_header=1).T
#label1 = 'dt = 0.1 Myr (first step split in 2)'''

#0 T
#1 d(x_HII)
#2 Gamma*dt
#3 Alpha*dt
#4 Ratio	
#5 x_HII
#6 Gamma*T
#7 Alpha*T
#8 Ratio
#9 Flux/S0

## Change
fig, axs = plt.subplots(2, 1, figsize=(6,6))
#axs[0].plot(data[0], data[1], data1[0], data1[1])
axs[0].plot(data[0], data[1], c='C0', label=r'$\Delta x_n$')
axs[0].plot(data[0], data[2], c='C1', label=r'$\Gamma\cdot dt$')
axs[0].plot(data[0], data[3], c='C2', label=r'$\alpha\cdot dt$')

axs[0].plot(data1[0], data1[1], c='C0', ls='--')
axs[0].plot(data1[0], data1[2], c='C1', ls='--')
axs[0].plot(data1[0], data1[3], c='C2', ls='--')

axs[0].legend(loc=4)
#axs[0].semilogy()
axs[0].set_title("Change in values over time step")

axs[1].plot(data[0], data[4], c='C0', label=label)

axs[1].plot(data1[0], data1[4], c='C0', label=label1, ls='--')

axs[1].legend(loc=1)
axs[1].semilogy()
axs[1].set_title("Conservation Ratio")
axs[1].set_xlabel("t (Myr)")
#axs[0].set_xlim(0, 2)
#axs[0].set_xlabel('time')
#axs[0].grid(True)

fig.tight_layout()
plt.savefig("./profiles/cons_delta.png")
plt.show()
sys.exit()
##################################################

fig, axs = plt.subplots(2, 1, figsize=(6,6))
#axs[0].plot(data[0], data[1], data1[0], data1[1])
axs[0].plot(data[0], data[5], c='C0', label=r'$\Delta x_n$')
axs[0].plot(data[0], data[6], c='C1', label=r'$\Gamma\cdot dt$')
axs[0].plot(data[0], data[7], c='C2', label=r'$\alpha\cdot dt$')

axs[0].plot(data1[0], data1[5], c='C0', ls='--')
axs[0].plot(data1[0], data1[6], c='C1', ls='--')
axs[0].plot(data1[0], data1[7], c='C2', ls='--')

axs[0].legend(loc=4)
#axs[0].semilogy()
axs[0].set_title("Change in values cumulatively")

axs[1].plot(data[0], data[8], c='C0', label=label)

axs[1].plot(data1[0], data1[8], c='C0', label=label1, ls='--')

axs[1].legend(loc=1)
axs[1].semilogy()
axs[1].set_title("Conservation Ratio")
axs[1].set_xlabel("t (Myr)")
#axs[0].set_xlim(0, 2)
#axs[0].set_xlabel('time')
#axs[0].grid(True)

fig.tight_layout()
plt.savefig("./profiles/cons_cumul.png")
plt.show()