#logical operators  = evaluate the true or false
#                     or = at least one caoditon is true
#                     and = all conditons are true
#                     not = reverse the results given in the program or we can say that i can invert the situation

# temp = float(input("enter the temperature: "))
# is_raining = bool(input("is it raining (yes or no) :"))

# if temp > 40 and is_raining == False:
#     print("it is hot and not raining")
# elif temp > 40 and is_raining == True:
#     print("it is hot and raining")
# elif temp < 40 and is_raining == False:
#     print("it is cold and not raining")
# else:
#     print("it is cold and raining")


# pin = 1234
# balance = 10000
# daily_limit = 50000
# account_status = "active"

# entered_pin = int(input("Enter your PIN: "))
# amount = int(input("Enter withdrawal amount: "))

# if entered_pin == pin and account_status and amount > 0 and amount <= balance and amount <= daily_limit:
#     print("Withdrawal successful!")
#     balance -= amount

# elif entered_pin != pin or not account_status:
#     print("Invalid PIN or inactive account.")

# elif amount <= 0 or amount > balance or amount > daily_limit:
#     print("Invalid withdrawal request.")

# else:
#     print("Transaction failed.")

# print("Remaining Balance:", balance)


#CONDITONAL STATEMENTS = A ONE LINE SHORTCUT FOR THE IF-ELSE STATEMENT
#                       PRINT OR ASSIGN ONE OF TWO VALUES BASED ON a  CONDITION


# num = int(input("enter a number:"))
# chekker = "even" if num % 2 == 0 else "odd"
# print(chekker)

# user_role = input("enter your role (admin/user) :")
# access_level = "Full acess" if user_role == "admin" else "limited access"

# print("Access level:", access_level)










# name  = input("name :")
# length = len(name)
# letter_find = name.find(input("enter a letter to find:"))      # first occurence of the letter will be found
# letter_rfind = name.rfind(input("enter a letter to find:"))    # last occurence of the letter will be found
# capital_letter = name.capitalize()  # first letter of the string will be capitalized
# full_capital_letter = name.upper()  # all letters of the string will be capitalized
# lower_case_letter = name.lower()
# digit_checker = name.isdigit()

phone_number = input("enter your phone number:")
result = phone_number.count("9")
print(result)




# print("length of name is:", length)
# print("letter found at index:", letter_find)
# print("letter found at index:", letter_rfind)
# print("capitalized name is:", capital_letter)
# print("full capitalized name is:", full_capital_letter)
# print("lower case name is:", lower_case_letter)
# print("is the digit: ", digit_checker)