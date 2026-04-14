print("example 1")
load = [120, 150, 100]
print("Total Load =", sum(load), "kW")
print("Maximum Load =", max(load), "kW")
print("Minimum Load =", min(load), "kW")
print("Number of Buses =", len(load))


print("example 2")
voltage_readings = [220.0, 225.5, 230.0, 228.0, 224.5]
# Basic analysis using built-in functions
number_of_readings = len(voltage_readings)
maximum_voltage = max(voltage_readings)
minimum_voltage = min(voltage_readings)
average_voltage = sum(voltage_readings) / len(voltage_readings)
sorted_voltage = sorted(voltage_readings)
# Display results
print("Voltage Readings:", voltage_readings)
print("Number of Readings:", number_of_readings)
print("Maximum Voltage:", maximum_voltage, "V")
print("Minimum Voltage:", minimum_voltage, "V")
print("Average Voltage:", average_voltage, "V")
print("Sorted Voltage Readings:", sorted_voltage)

print("example 3")
# System information
feeder_name = "Feeder B" # string
operating_hours = 6 # integer (hours)
tariff = 6.0 # float (Rs per kWh)
# Measured values
voltage = [220.0, 225.0, 230.0] # list of floats (Volts)
current = [4.8, 5.0, 5.2] # list of floats (Amperes)
# Power calculation (P = V × I)
power_1 = voltage[0] * current[0]
power_2 = voltage[1] * current[1]
power_3 = voltage[2] * current[2]
power = [power_1, power_2, power_3] # list of power values
# Average power
average_power = sum(power) / len(power)
# Energy calculation (convert W to kWh)
energy_consumed = (average_power * operating_hours) / 1000
# Cost calculation
total_cost = energy_consumed * tariff
# Display results
print("Feeder Name:", feeder_name)
print("Voltage Readings (V):", voltage)
print("Current Readings (A):", current)
print("Power Readings (W):", power)
print("Average Power (W):", average_power)
print("Energy Consumed (kWh):", energy_consumed)
print("Total Electricity Cost (Rs):", total_cost)


print("Example")
# STRING: System information
system_name = "Feeder A"
# INTEGER: Number of operating hours
operating_hours = 5
# TUPLE: Fixed rated values (cannot change)
rated_values = (400, 50) # Voltage (V), Frequency (Hz)
# LIST: Measured voltage readings (can change)
voltage_readings = [220.0, 225.5, 230.0, 228.0]
# DICTIONARY: Bus-wise load data (labeled data)
bus_load = {
"Bus1": 120,
"Bus2": 150,
"Bus3": 100
}
# --- Display basic information ---
print("System Name:", system_name)
print("Operating Hours:", operating_hours)
# --- Accessing tuple values using index ---
print("Rated Voltage:", rated_values[0], "V")
print("Rated Frequency:", rated_values[1], "Hz")
# --- List analysis using inbuilt functions ---
print("Voltage Readings:", voltage_readings)
print("Number of Voltage Readings:", len(voltage_readings))
print("Maximum Voltage:", max(voltage_readings), "V")
print("Minimum Voltage:", min(voltage_readings), "V")
print("Average Voltage:", sum(voltage_readings) / len(voltage_readings), "V")
print("Sorted Voltage Readings:", sorted(voltage_readings))
# --- Dictionary analysis ---
print("Bus-wise Load Data:", bus_load)
print("Number of Buses:", len(bus_load))
print("Maximum Load:", max(bus_load.values()), "kW")
print("Minimum Load:", min(bus_load.values()), "kW")