#1 Printing Output

print("Welcome to Python Programming Lab")
print("This is Practical 1")

#2 Variables and Arithmetic Operations

a = 10
b = 5
sum_value = a + b
difference = a - b
product = a * b
division = a / b
print("Sum =", sum_value)
print("Difference =", difference)
print("Product =", product)
print("Division =", division)

#User Input and Type Conversion

x = input("Enter first number: ")
y = input("Enter second number: ")
x = float(x)
y = float(y)
average = (x + y) / 2
print("Average =", average)

# 4 Simple Engineering Example

voltage = float(input("Enter voltage (V): "))
current = float(input("Enter current (A): "))
power = voltage * current
print("Electrical Power =", power, "Watts")

'''Electrical Energy Consumption Calculation'''

device_name = "Electric Heater" # string
voltage = 230.0 # float (Volts)
current = 4.5 # float (Amperes)
hours = 5 # integer (hours)
power = voltage * current # Power in Watts
energy = power * hours # Energy in Watt-hours
print("Device Name:", device_name)
print("Power =", power, "Watts")
print("Energy Consumed =", energy, "Watt-hours")

