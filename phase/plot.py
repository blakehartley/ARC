import sys
import numpy as np
import matplotlib.pyplot as plt

array_dict = {}

array_dict['base'] = ["./data/output_base_10.npy", "Hubble Cooling, Adaptive dt"]
array_dict['reduced'] = ["./data/output_reduced_10.npy", "Reduced, Adaptive dt"]
array_dict['shifted'] = ["data/output_shifted_10.npy", "Shifted, Adaptive dt"]
array_dict['helium'] = ["data/output_helium_10.npy", "Helium, Adaptive dt"]
array_dict['lambda'] = ["data/output_lambda_10.npy", "Lambda, Adaptive dt"]

for key, array in array_dict.items():
    data = np.load(array[0])
    label = array[1]
    print(label)
    #print(data[0], data[1])
    plt.plot(data[1], data[0], label=label)

    #plt.axis([1.e1,3.e5,1.e0,1e8])
    plt.legend(loc=1)
    plt.loglog()
    #plt.title(r'${\rm Hydrogen}\;{\rm Fraction}$', fontsize=24)
plt.savefig("phase_combined.png")
plt.close()