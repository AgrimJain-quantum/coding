import numpy as np
random_numbers = np.random.rand(5)
print(random_numbers)

import numpy as np
voltage = np.random.uniform(220, 240, 5)
print(voltage)

import numpy as np
# number of simulations
N = 1000
# random inputs
voltage = np.random.uniform(220, 240, N)
current = np.random.uniform(4, 6, N)
# power calculation
power = voltage * current
print("Sample Power Values:", power[:10])
print("Average Power:", np.mean(power))
print("Maximum Power:", np.max(power))
print("Minimum Power:", np.min(power))

import matplotlib.pyplot as plt 
plt.hist(power, bins=20) 
plt.title("Power Distribution (Monte Carlo)") 
plt.xlabel("Power (W)") 
plt.ylabel("Frequency") 
plt.show()