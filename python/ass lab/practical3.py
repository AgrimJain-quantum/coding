# ==============================
# Experiment 3: Control Structures & Functions
# ==============================

# -------- Conditional Statements --------

voltage = 235
if voltage > 230:
    print("Voltage is above safe limit")
else:
    print("Voltage is within safe limit")

voltage = 215
if voltage < 220:
    print("Low Voltage Condition")
elif voltage > 230:
    print("High Voltage Condition")
else:
    print("Voltage within normal range")

# -------- Loops --------

voltages = [220, 225, 230, 228]
for v in voltages:
    print("Voltage reading:", v)

voltage = [220, 225, 230]
current = [5, 5.2, 4.8]
for i in range(3):
    power = voltage[i] * current[i]
    print("Power:", power, "W")

voltage = [220, 225, 230]
current = [5, 5.2, 4.8]
for v, i in zip(voltage, current):
    power = v * i
    print("Power:", power, "W")

numbers = list(range(5))
print(numbers)

numbers = list(range(2, 6))
print(numbers)

voltage = [220, 225, 230]
current = [5, 5.2, 4.8]
pairs = list(zip(voltage, current))
print(pairs)

voltages = [220, 225, 230]
total = 0
for v in voltages:
    total = total + v
print("Total voltage:", total)

# -------- Functions --------

def calculate_power(v, i):
    return v * i

def energy(power, time):
    return power * time

def average_voltage(voltage_list):
    return sum(voltage_list) / len(voltage_list)

def maximum_voltage(v_list):
    return max(v_list)

# -------- Main Execution --------

voltages = [220, 225, 230]
currents = [5, 5.2, 4.8]

print("Power Readings:")
for k in range(len(voltages)):
    p = calculate_power(voltages[k], currents[k])
    print("Power:", p, "W")

avg_v = average_voltage(voltages)
print("Average Voltage:", avg_v)

max_v = maximum_voltage(voltages)
print("Maximum Voltage:", max_v)

E = energy(1000, 5)
print("Energy:", E, "Wh")