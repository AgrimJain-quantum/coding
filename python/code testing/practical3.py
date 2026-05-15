# A. Store system information
feeder_name = "Feeder A"
rated_voltage = 400
rated_frequency = 50

voltage = [220, 225, 230]
current = [5, 5.2, 4.8]

operating_time = 4

bus_load = {
    "Bus1": 120,
    "Bus2": 150,
    "Bus3": 100
}

# B & C. Calculate power using P = V * I
power = []
for i in range(len(voltage)):
    p = voltage[i] * current[i]
    power.append(p)
    print("Power:", p, "W")

# D. Average voltage
avg_voltage = sum(voltage) / len(voltage)
print("Average Voltage =", avg_voltage, "V")

# E. Total energy consumption
average_power = sum(power) / len(power)
energy = average_power * operating_time
print("Total Energy =", energy, "Wh")

# F. Max and Min voltage
print("Maximum Voltage =", max(voltage))
print("Minimum Voltage =", min(voltage))

# G. Display bus loads using loop
for bus in bus_load:
    print(bus, "=", bus_load[bus], "kW")