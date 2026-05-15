# Printing
print("Welcome to Python Programming Lab")
print("This is Practical 1")

# Arithmetic Operations
a = 10
b = 5
print("Sum =", a + b)
print("Difference =", a - b)
print("Product =", a * b)
print("Division =", a / b)

# User Input
x = float(input("Enter first number: "))
y = float(input("Enter second number: "))
average = (x + y) / 2
print("Average =", average)

# Power Calculation
voltage = float(input("Enter voltage (V): "))
current = float(input("Enter current (A): "))
power = voltage * current
print("Electrical Power =", power, "Watts")

# Energy Calculation
device_name = "Electric Heater"
voltage = 230.0
current = 4.5
hours = 5
power = voltage * current
energy = power * hours

print("Device Name:", device_name)
print("Power =", power, "Watts")
print("Energy Consumed =", energy, "Wh")