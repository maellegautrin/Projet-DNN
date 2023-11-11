#!/usr/bin/env python3

from z3 import *

solver = Solver()

s = [784,200,100,60,30]
L = []

layers = []
layers_after_relu = []

y = []
x = []
b = []
W = []



def init_vars():
    x = [ [ Float(f"x_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    y = [ [ Float(f"y_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    b = [ [ Float(f"b_{j}_{i}") for j in range(s[i]) ] for i in range(n) ]
    W = [ [ [Float(f"w_{j}_{i}_{k}") for k in range(s[i-1])] for j in range(s[i]) ] for i in range(1,n) ]

init_vars()

def get_contraint(N,x_star, L,j):
    def eq_layer(x,y)=
        And([x[i] == y[i] for i in range(s[0])])


    def c_in(x):
        eq_layer(x,layers[0])

    def matrice_product(x,y):
        temp = 0
        for i in range(len(x)):
            temp = temp + x[i] * y[i]

    def c(i):
        And([ y[i][j] == matrice_product(W[i-1][j],x[i - 1]) + b[i][j] for j in range(s[i]) ])


    def c_prime(i):
        And( [  And(Implies(y[i][j] > 0, x[i][j] == y[i][j]), Implies(y[i][j] <= 0, x[i][j] == 0) )  for j in range(s[i]) ] )

    def c_out(L,j):
        And([  x[n][k] <= x[n][j]  for k in range(s[-1]) if k != j ])

    temp = [And(c(i),c_prime(i)) for i in range(1,n+1)]
    temp.append(c_in(x_star))
    temp.append(c_out(L,j))
    And(temp)



def Max(x,y):
    If(x > y, x, y)

def distance(x,y):
    temp = 0
    for i = 0 in range(len(x)):
        temp = Max(temp,Abs(x[i] - y[i]))
    temp


def find_epsilon() =
    epsilon = -0.1
    is_sat = false

    solver.add(get_contrain(N,x,L_star,j))

    while !is_sat:
        print(f"test: {epsilon}")
        epsilon += 0.1
        solver.add(distance(x_star,x) < epsilon)
        is_sat = solver.sat() == sat
        if !is_sat:
            solver.pop()
    print(f"find epsilon : {epsilon}")
