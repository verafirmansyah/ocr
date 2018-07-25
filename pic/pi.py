#!/usr/bin/env python
# encoding: utf-8

# pi.png 的标签
#number = 2103

# pi_01.png 的标签
number = 224

number1 = number + 10
b = 10 ** number1
x1 = b * 4 // 5
x2 = b // -239
he = x1 + x2
number *= 2
for i in xrange(3, number, 2):
    x1 //= -25
    x2 //= -57121
    x = (x1 + x2) // i
    he += x

pai = he * 4
pai //= 10**10
paistr = str(pai)
result = paistr[0] + '.' + paistr[1:len(paistr)]
print(result)


