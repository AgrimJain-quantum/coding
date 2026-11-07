import math
# print ("hello world")

# # variable = a container for a value( string , integer , float , boolean). it behaves as if it was the value that it contains 

# #strings
# first_name = "agrim jain"
# food = "i like pizza"
# email = "agrimjain@example.com"


# print(first_name)
# print(f"Hello, {first_name}!")    #f-strings 
# print(f"hi! , {food}")
# print(f"Your email is: {email}")

# # integers 
# age = 21
# quantity = 5

# print(f"my age is {age}")
# print(f"i have {quantity} items")

# #floats
# height = 5.9
# weight = 70.5

# print(f"my height is {height} feet")
# print(f"my weight is {weight} kg")

# #booleans
# is_student = True
# if is_student:
#     print("I am a student")
# else:
#     print("I am not a student")
    
# for_sale = True
# if for_sale:
#     print("This item is for sale")
# else:
#     print("This item is not for sale")
    
# name = input("Enter your name: ")
# print(f"Hello, {name}!")

## typecasting  =  converting the data type of a value to another data type
# name = input("Enter your name: ")
# name = bool(name)
# print(name)

# age = 25
# print(f"My age is {age}")
# print(type(age))  # Output: <class 'int'>

#input() = a fucntion that prompts the user to enter data
#          returns the entered data as a string
# name = input("Enter your name:")
# print(f"Hello, {name}!")


# age = input("Enter your age: ")
# age = int(age)  # Convert the input string to an integer
# age += 1

# print(f"happy birthday! you are now {age} years old")  # Use f-string for concatenation

# year = int(input("Enter the current year: "))
# print(f"Next year will be {year + 1}")  # Use f-string for concatenation

# # exercise 1
# # find the area of a rectangle

# l = float(input("Enter the length of the rectangle: "))
# w = float(input("Enter the width of the rectangle: "))
# area = l * w
# print(f"The area of the rectangle is {area} cm²")

# # exercise 2 shopping cart 

# item  = input("Enter the item you want to buy: ")
# price = float(input(f"Enter the price of {item}: "))
# quantity = int(input(f"Enter the quantity of {item}: "))

# total = price * quantity
# print(f"The total cost of {quantity} {item}(s) is ${total:.2f}")  # Format total to 2 decimal places

# #subscripting
# name = "agrim jain"
# print(name[0])  # Output: a

# SOME OTHER AIRTHEMETIC OPERATOR 

x = 3.14
y = 2
z = 5
l = 64.345678
k = 5.6

# result = round(x)  # Round x to the nearest integer
# result = round(x, 1)  # Round x to 1 decimal place
# result = abs(x)
# result = pow(y, z)  # Calculate y raised to the power of z (y^z)
# result = min(x , y, z)
# result = max(x , y, z)
# result = math.sqrt(z)  # Calculate the square root of z
# result_1 = math.ceil(l)  # Calculate the ceiling of l
# result_2 = math.floor(k)
# print(result)
# print(result_1)
# print(result_2)


# # perimeter of circle = 2 * math.pi * r
# radius = float(input("enter the radius of the circle:"))
# p = 2 * math.pi * radius
# print(f"the perimeter is {round(p)}")

# # hypotenuse calculation 

# a = float(input("enter the lenght of side a:"))
# b = float(input("enter the lenght of side b:"))

# c = math.sqrt(pow(a, 2) + pow(b, 2))
# print(f"the hypotenuse is {round(c)}")

# age = int(input("enter your age:"))
# if age >= 18:
#     print("you are eligible to vote")
# elif age < 0:
#     print("invalid age")
# else:
#     print("you are not eligible to vote")
    
# for_sale = bool(input("is the item for sale? (True/False): "))

# if for_sale == True:
#     print("This item is for sale")
# elif for_sale == False:
#     print("This item is not for sale")
# else:
#     print("out of stock")
    
# python calculator 
operator = input("enter the operator (+, -, *, /):")
n1 = float(input("enter the first no. :  "))
n2 = float(input("enter the second no. :"))

if operator == "+":
    result = n1 + n2
    print(f"the result is {result}")
elif operator == "-":
    result = n1 - n2
    print(f"the result is {result}")
elif operator == "*":
    result = n1 * n2
    print(f"the result is {result}")
elif operator == "/":
    if n2 != 0:
        result = n1 / n2
        print(f"the result is {result}")
    else:
        print("Error: Division by zero is not allowed.")
    
    





