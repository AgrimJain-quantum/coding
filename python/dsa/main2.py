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


pin = 1234
balance = 10000
daily_limit = 50000
account_status = "active"

entered_pin = int(input("Enter your PIN: "))
amount = int(input("Enter withdrawal amount: "))

if entered_pin == pin and account_status and amount > 0 and amount <= balance and amount <= daily_limit:
    print("Withdrawal successful!")
    balance -= amount

elif entered_pin != pin or not account_status:
    print("Invalid PIN or inactive account.")

elif amount <= 0 or amount > balance or amount > daily_limit:
    print("Invalid withdrawal request.")

else:
    print("Transaction failed.")

print("Remaining Balance:", balance)


