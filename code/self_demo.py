# # Method 1: direct operation.
# # Problem: need to change the code if want the numbers to change, cannot reuse the code
# # This is facing towards process
# num1 = 100
# num2 = 20
# res = num1 + num2
# print(f"Sum = {res}")
#
# # Method 2: functions
# # Problem: different functions cannot use each other's properties (ex: variables)
# def add(num1, num2): # Doesn't need to specify data type
#     return num1+num2
# print(f"Sum = {add(90,20)}")
# print(f"Sum = {add(290,20)}")

# Method 3: class
# If inside class, self. something can be used throughout class
# Is an alternative to global variables
# Advantage: prevents overwriting of variables
# Useful for team projects when the coding is split up (prevent conflicting variable names)
# Cannot just break into different programs since variables will still conflict that way
class AddSub:
    def __init__(self): # override the initialization program
        # here, can also use super() to inherit properties from higher hierarchies
        self.res = 0

    def add(self, num1, num2):
        self.res = num1 + num2
        return self.res #can modify res, then use res

    def sub(self, num3):
        return self.res - num3 #or give a return value and directly use the function

name = AddSub() # initialize the class using __init__. Also 实例化 the class: what is it?
print(name.res) # just means variable res inside class name
name.add(3,5)
print(name.res)

print(name.sub(2))

