from calculator import Calculator

cal = Calculator()

assert cal("4*3-3") == 9
assert cal("(2+3)*4") == 20
assert cal("10/2+7") == 12
assert cal("18-(3*4)") == 6
assert cal(" 8 + 2 * 5 ") == 18

print("calculator tests passed")
