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


age = input("Enter your age: ")
age = int(age)  # Convert the input string to an integer
age += 1

print(f"happy birthday! you are now {age} years old")  # Use f-string for concatenation

year = int(input("Enter the current year: "))
print(f"Next year will be {year + 1}")  # Use f-string for concatenation

# ecercise 1
# find the area of a rectangle

l = float(input("Enter the length of the rectangle: "))
w = float(input("Enter the width of the rectangle: "))
area = l * w
print(f"The area of the rectangle is {area}")
