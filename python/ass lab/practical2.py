bus_numbers = [1, 2, 3, 4]

voltage = [220.0, 225.5, 230.0]

devices = ["Fan", "Light", "Heater"]

data = [1, 230.5, "Feeder A"]

voltage = [220, 225, 230, 228]
print("Number of readings =", len(voltage))

power = [100, 150, 200]
total_power = sum(power)
print("Total Power =", total_power, "W")

voltage = [220, 225, 230, 228]
print("Maximum Voltage =", max(voltage))
print("Minimum Voltage =", min(voltage))

voltage = [230, 220, 228, 225]
sorted_voltage = sorted(voltage)
print("Original list:", voltage)
print("Sorted list:", sorted_voltage)

voltage = [220, 225, 230, 225, 228]
print("225 V appears", voltage.count(225), "times")

voltage = [220, 225, 230, 228]
position = voltage.index(230)
print("230 V is at index", position)

print(voltage[0])
print(voltage[1])
print(voltage[2])

print(voltage[-1])

voltage[1] = 227
print(voltage)

load = [120, 150, 100]
print("Total Load =", sum(load), "kW")
print("Maximum Load =", max(load), "kW")
print("Minimum Load =", min(load), "kW")
print("Number of Buses =", len(load))

voltage_readings = [220.0, 225.5, 230.0, 228.0, 224.5]

number_of_readings = len(voltage_readings)
maximum_voltage = max(voltage_readings)
minimum_voltage = min(voltage_readings)
average_voltage = sum(voltage_readings) / len(voltage_readings)
sorted_voltage = sorted(voltage_readings)

print("Voltage Readings:", voltage_readings)
print("Number of Readings:", number_of_readings)
print("Maximum Voltage:", maximum_voltage, "V")
print("Minimum Voltage:", minimum_voltage, "V")
print("Average Voltage:", average_voltage, "V")
print("Sorted Voltage Readings:", sorted_voltage)

feeder_name = "Feeder B"
operating_hours = 6
tariff = 6.0

voltage = [220.0, 225.0, 230.0]
current = [4.8, 5.0, 5.2]

power_1 = voltage[0] * current[0]
power_2 = voltage[1] * current[1]
power_3 = voltage[2] * current[2]
power = [power_1, power_2, power_3]

average_power = sum(power) / len(power)

energy_consumed = (average_power * operating_hours) / 1000

total_cost = energy_consumed * tariff

print("Feeder Name:", feeder_name)
print("Voltage Readings (V):", voltage)
print("Current Readings (A):", current)
print("Power Readings (W):", power)
print("Average Power (W):", average_power)
print("Energy Consed (kWh):", energy_consumed)
print("Total Electricity Cost (Rs):", total_cost)

rated_voltage = (220.0, 225.0, 230.0)

print("Rated Voltage Values:", rated_voltage)

print("First rated voltage:", rated_voltage[0], "V")
print("Last rated voltage:", rated_voltage[-1], "V")

print("Number of values:", len(rated_voltage))
print("Maximum rated voltage:", max(rated_voltage), "V")
print("Minimum rated voltage:", min(rated_voltage), "V")
print("Total of rated voltages:", sum(rated_voltage))

sorted_voltage = sorted(rated_voltage)
print("Sorted voltage values:", sorted_voltage)

bus_load = {
 "Bus1": 120,
 "Bus2": 150,
 "Bus3": 100
}

print("Load at Bus1:", bus_load["Bus1"], "kW")
print("Load at Bus2:", bus_load["Bus2"], "kW")

bus_load["Bus4"] = 130
print(bus_load)

bus_load["Bus2"] = 155
print(bus_load)

print("Bus3" in bus_load)

print(len(bus_load))

print(bus_load.values())

print(bus_load.items())

print("Maximum Load:", max(bus_load.values()), "kW")
print("Minimum Load:", min(bus_load.values()), "kW")

system_name = "Feeder A"

operating_hours = 5

rated_values = (400, 50)

voltage_readings = [220.0, 225.5, 230.0, 228.0]

bus_load = {
 "Bus1": 120,
 "Bus2": 150,
 "Bus3": 100
}

print("System Name:", system_name)
print("Operating Hours:", operating_hours)

print("Rated Voltage:", rated_values[0], "V")
print("Rated Frequency:", rated_values[1], "Hz")

print("Voltage Readings:", voltage_readings)
print("Number of Voltage Readings:", len(voltage_readings))
print("Maximum Voltage:", max(voltage_readings), "V")
print("Minimum Voltage:", min(voltage_readings), "V")
print("Average Voltage:", sum(voltage_readings) / len(voltage_readings), "V")
print("Sorted Voltage Readings:", sorted(voltage_readings))

print("Bus-wise Load Data:", bus_load)
print("Number of Buses:", len(bus_load))
print("Maximum Load:", max(bus_load.values()), "kW")
print("Minimum Load:", min(bus_load.values()), "kW")