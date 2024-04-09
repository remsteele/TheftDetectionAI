
l1, l2, carry = 7, 4, 2

value = l1 + l2 + carry
print(value)
value, carry = value % 10, value // 10

print(value, carry)