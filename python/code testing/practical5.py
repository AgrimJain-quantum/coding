import matplotlib.pyplot as plt
voltage = [220, 225, 230, 228]
plt.plot(voltage)
plt.show()

import matplotlib.pyplot as plt
voltage = [220, 225, 230, 228]
plt.plot(voltage)
plt.title("Voltage Readings")
plt.xlabel("Reading Number")
plt.ylabel("Voltage (V)")
plt.show()

import matplotlib.pyplot as plt
voltage = [220, 225, 230]
current = [5, 5.2, 4.8]
plt.plot(voltage, current)
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Voltage vs Current")
plt.show()

import matplotlib.pyplot as plt
voltage = [220, 225, 230]
current = [5, 5.2, 4.8]
plt.scatter(voltage, current)
plt.title("Scatter Plot")
plt.xlabel("Voltage")
plt.ylabel("Current")
plt.show()