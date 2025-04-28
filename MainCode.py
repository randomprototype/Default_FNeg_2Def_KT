import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from scipy.integrate import quad #Single integral
from scipy.integrate import dblquad
from PIL import Image

###Defining the p.d.f.s and reliability functions############################
def fx(x):
    return (b1 / a1**b1) * (x**(b1 - 1)) * np.exp(-(x / a1)**b1)
def fy(y):
    return (b2 / a2**b2) * (y**(b2 - 1)) * np.exp(-(y / a2)**b2)
def fh(h):
    return (b3 / a3**b3) * (h**(b3 - 1)) * np.exp(-(h / a3)**b3)
def Rx(x):
    return 1 - (1 - np.exp(-(x / a1)**b1))
def Ry(y):
    return 1 - (1 - np.exp(-(y / a2)**b2))
##############################################################################

###Defining the expected inspection cost######################################
def CompExpCost(i):
    Sum=0
    for j in range(1,i,1):
        Sum+=binom.pmf(j, i-1, (1-p))*j*Ci
    return Sum

###Defining all the scenarios################################################
def P1(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K+1):
        Pi = ((1-q)**(i-1)) * tplquad(lambda h, y, x: fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
        prob += Pi
        length += ((1-q)**(i-1)) * tplquad(lambda h, y, x: (x + y + h) * fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
        cost += Pi * (Cf + CompExpCost(i))+((1-q)**(i-1)) * tplquad(lambda h, y, x: (Cmd*h)*fx(x) * fy(y) * fh(h), (i-1)*T, i*T, 0, lambda x: (i*T) - x, 0, lambda x, y: (i*T) - x - y)[0]
    return prob, length, cost
def P2(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K):
        pi = ((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
        prob += pi
        length += ((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: (i*T + y + h) * fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
        cost += pi * (Cf + Ci + CompExpCost(i))+((1-q)**(i-1)) * q * Rx(i*T) * dblquad(lambda h, y: (Cmd*h)*fy(y) * fh(h), 0, T, 0, lambda y: T - y)[0]
    return prob, length, cost
def P3(K, T):
    prob, length, cost = 0, 0, 0
    for i in range(1, K):
        for j in range(i+1, K+1):
            pi = ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
            prob += pi
            length += ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: (x + y + h) * fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
            cost += pi * (Cf + CompExpCost(j)) + ((1-q)**(i-1)) * ((p + (1-p) * beta)**(j-i)) * tplquad(lambda h, y, x: (Cmd*h)*fx(x) * fy(y) * fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T) - x, lambda x: (j*T) - x, 0, lambda x, y: (j*T) - x - y)[0]
    return prob, length, cost
def P4(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+2,K+1):
            pi = ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
            prob += pi
            length += ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
            cost += pi *(Cf + Ci + CompExpCost(j-1)) + ((q*(1-q)**(i-1))*((p+(1-p)*beta)**(j-i-1))*Rx(i*T)*dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), ((j-1)-i)*T, (j-i)*T, 0, lambda y: j*T-i*T-y)[0]) 
    return prob, length, cost
def P5(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        for j in range (i,K):
            pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(1-beta)*(1-p)*(dblquad(lambda y, x: fx(x)*fy(y), (i-1)*T, i*T, lambda x: (j*T)-x, np.inf))[0]
            prob += pi
            length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(1-beta)*(1-p)*(dblquad(lambda y, x: j*T*fx(x)*fy(y), (i-1)*T, i*T, lambda x: (j*T)-x, np.inf))[0]
            cost += pi*(Cr + Ci + CompExpCost(j))
    return prob, length, cost
def P6(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+1,K):
            pi = (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(1-beta)*(1-p)*Rx(i*T))*quad(lambda y: fy(y), (j-i)*T, np.inf)[0]) 
            prob += pi
            length += (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(1-beta)*(1-p)*Rx(i*T))*quad(lambda y: j*T*fy(y), (j-i)*T, np.inf)[0]) 
            cost += pi * (Cr + 2*Ci + CompExpCost(j-1))
    return prob, length, cost
def P7(K,T):
    prob,  length, cost = 0, 0, 0
    for i in range (1,K):
        for j in range (i,K):
            pi = (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
            prob += pi
            length += (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: j*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
            cost += pi * (Cr + Ci + CompExpCost(j)) + (((1-q)**(i-1))*((p)**(j-i))*(1-p)*(tplquad(lambda h, y, x: (Cmd*(i*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: i*T-x, lambda x, y: (((j)*T)-x-y), np.inf))[0])
    return prob, length, cost
def P8(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+1,K):
           pi = ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
           prob += pi
           length += ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: j*T*fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
           cost += pi * ((Cr + 2*Ci + CompExpCost(j-1))) + ((((1-q)**(i-1))*q*(p**(j-i-1))*(1-p)*Rx(i*T))*dblquad(lambda h, y: (Cmd*(j*T-(i*T+y)))*fy(y)*fh(h), 0, T, lambda y: j*T-y-i*T, np.inf)[0]) 
   return prob , length , cost
def P9(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-1):
        for j in range (i+1,K):
            for l in range (j,K):
                pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
                prob += pi
                length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: l*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
                cost += pi * (Cr + Ci + CompExpCost(l)) + ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(1-p)*(tplquad(lambda h, y, x: (Cmd*(l*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: ((l*T)-x-y), np.inf))[0]
    return prob, length, cost
def P10(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-2):
        for j in range (i+2,K):
            for l in range (j,K):
                pi = ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
                prob += pi
                length += ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: l*T*fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
                cost += pi * ((Cr + 2*Ci + CompExpCost(l-1))) + ((q*(1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*(1-p)*Rx((i)*T))*dblquad(lambda h, y: (Cmd*(l*T-(i*T+y)))*fy(y)*fh(h), (j-1)*T-i*T, j*T - (i)*T, lambda y: (l*T)-y-(i)*T, np.inf) [0]) 
    return prob, length, cost
def P11(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K+1):
        pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(K-i))*(dblquad(lambda y, x: fx(x)*fy(y), (i-1)*T, i*T, lambda x: (K*T)-x, np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*((p+(1-p)*beta)**(K-i))*(dblquad(lambda y, x: K*T*fx(x)*fy(y), (i-1)*T, i*T, lambda x: (K*T)-x, np.inf))[0]
        cost += pi * (Cr + CompExpCost(K))
    return prob, length, cost
def P12(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        pi = (q*(1-q)**(i-1))*((p+(1-p)*beta)**(K-i-1))*Rx((i)*T)*(quad(lambda y: fy(y), (K*T)-(i)*T, np.inf))[0]
        prob += pi
        length += (q*(1-q)**(i-1))*((p+(1-p)*beta)**(K-i-1))*Rx((i)*T)*(quad(lambda y: K*T*fy(y), (K*T)-(i)*T, np.inf))[0]
        cost += pi * (Cr + Ci + CompExpCost(K-1))
    return prob, length, cost
def P13(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K+1):
        pi = ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: K*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
        cost += pi * (Cr + CompExpCost(i)) + ((1-q)**(i-1))*(p**(K-i))*(tplquad(lambda h, y, x: (Cmd*(K*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]
    return prob, length, cost
def P14(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K):
        pi = ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
        prob += pi
        length += ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: K*T*fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
        cost += pi * (Cr + Ci + CompExpCost(i)) + ((1-q)**(i-1))*q*(p**(K-i-1))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*(K*T-(i*T+y)))*fy(y)*fh(h), 0, T, lambda y: (K*T)-y-i*T, np.inf))[0]
    return prob, length, cost
def P15(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K):
       for j in range (i+1,K+1):
           pi = (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
           prob += pi
           length += (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: K*T*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
           cost += pi * (Cr + CompExpCost(j)) + (((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(K-j))*(tplquad(lambda h, y, x: (Cmd*(K*T-(x+y)))*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: (j-1)*T - x, lambda x: (j*T)-x, lambda x, y: ((K*T)-x-y), np.inf))[0]) 
   return prob, length, cost
def P16(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+2,K+1):
           pi = ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
           prob += pi
           length += ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: K*T*fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
           cost += pi * (Cr + Ci + CompExpCost(j-1)) + ((1-q)**(i-1))*q*((p+(1-p)*beta)**(j-i-1))*(p**(K-j))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*(K*T-(i*T+y)))*fy(y)*fh(h), (j-1)*T-i*T, j*T-i*T, lambda y: (K*T)-y-i*T, np.inf))[0]
   return prob, length, cost 
def P17(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K):
       for j in range (i+1,K+1):
           pi = ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
           prob += pi
           length += ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: (x+y+h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
           cost += pi * (Cf + CompExpCost(i)) + ((1-q)**(i-1))*(p**(j-i))*(tplquad(lambda h, y, x: (Cmd*h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, 0, lambda x: (i*T)-x, lambda x, y: (((j-1)*T)-x-y), lambda x, y: ((j*T)-x-y)))[0] 
   return prob, length, cost 
def P18(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+2,K+1):
           pi = ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
           prob += pi
           length += ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
           cost += pi * (Cf + Ci + CompExpCost(i)) + ((1-q)**(i-1))*q*(p**(j-i-1))*Rx(i*T)*(dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), 0, T, lambda y: ((j-1)*T)-y-i*T,lambda y: (j*T)-y-i*T))[0]
   return prob, length, cost 
def P19(K,T):
   prob, length, cost = 0, 0, 0
   for i in range (1,K-1):
       for j in range (i+1,K):
           for l in range (j+1,K+1):
               pi = ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
               prob += pi
               length += ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: (x+y+h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
               cost += pi * (Cf + CompExpCost(j)) + ((1-q)**(i-1))*((p+(1-p)*beta)**(j-i))*(p**(l-j))*(tplquad(lambda h, y, x: (Cmd*h)*fx(x)*fy(y)*fh(h), (i-1)*T, i*T, lambda x: ((j-1)*T)-x, lambda x: (j*T)-x, lambda x, y: (((l-1)*T)-x-y), lambda x, y:((l*T)-x-y)))[0]
   return prob, length, cost 
def P20(K,T):
    prob, length, cost = 0, 0, 0
    for i in range (1,K-2):
        for j in range (i+2,K):
            for l in range (j+1,K+1):
                pi = (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
                prob += pi
                length += (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: (i*T+y+h)*fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
                cost += pi * (Cf + Ci + CompExpCost(j-1)) +  (q*((1-q)**(i-1))*(((p+(1-p)*beta)**(j-i-1))*(p**(l-j))*Rx(i*T))*dblquad(lambda h, y: (Cmd*h)*fy(y)*fh(h), (j-1)*T - i*T, j*T - i*T, lambda y: ((l-1)*T)-y-(i*T), lambda y:(l*T)-y-(i*T)) [0])
    return prob, length, cost 
def P21(K, T):
    prob = ((1-q)**(K-1)) * Rx(K*T)
    length = ((1-q)**(K-1)) * Rx(K*T) * K*T
    cost = prob * (Cr + CompExpCost(K))
    return prob, length, cost

def Sum(K, T):
    prob, length, cost = 0, 0, 0
    cenarios = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21]
    for cenario in cenarios:
        p, l, c = cenario(K, T)
        prob += p
        length += l
        cost += c
    return prob, length, cost

def taxa_de_custo(K, T):
    prob, length, cost = Sum(K, T)
    return cost/length

def main():
    #criando 3 colunas
    col1, col2, col3= st.columns(3)
    foto = Image.open('randomen.png')
    #st.sidebar.image("randomen.png", use_column_width=True)
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    #O código abaixo centraliza e atribui cor
    st.markdown("<h2 style='text-align: center; color: #306754;'>HIDF-2D: Hybrid Inspection/Age-based with Defaults, False-negative and 2 levels of Defect</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #F3F3F3; padding: 10px; text-align: center;">
          <p style="font-size: 20px; font-weight: bold;">Data-driven maintenance scheduling for a system with a three-stage failure process subject to maintenance errors and defaults</p>
          <p style="font-size: 15px;">By: Rafael G. N. Paiva, Victor H. R. Lima, Augusto J. S. Rodrigues, Cristiano A. V. Cavalcante & P. Do</p>
        </div>
        """, unsafe_allow_html=True)

    menu = ["Cost-rate", "Information", "Website"]
    
    choice = st.sidebar.selectbox("Select here", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insert the parameter values below:")
        
        global a2,b2,a1,b1,p,l,alfa,beta,ci,cr,cf,cd
        a2=st.number_input("Insert the scale parameter for the defect arrival distribution of “strong” components (η\u2081)", min_value = 0.0, value = 3.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the defect arrival for the stronger component.")
        b2=st.number_input("Insert the shape parameter for the defect arrival distribution of “strong” components (β\u2082)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the stronger component.")
        a1=st.number_input("Insert the scale parameter for the defect arrival distribution of “weak” components (η\u2081)", min_value = 3.0, value = 18.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        b1=st.number_input("Insert the shape parameter for the defect arrival distribution of “weak” components (β\u2082)", min_value = 1.0, max_value=5.0, value = 5.0, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        p=st.number_input("Insert the mixture parameter (p)", min_value = 0.0, max_value=1.0, value = 0.10, help="This parameter indicates the proportion of the weaker component within the total population of components.")
        l=st.number_input("Insert the rate of the exponential distribution for delay-time (λ)", min_value = 0.0, value = 2.0, help="This parameter defines the rate of the Exponential distribution, which governs the transition from the defective to the failed state of a component.")
        alfa=st.number_input("Insert the false-positive probability (\u03B1)", min_value = 0.0, max_value=1.0, value = 0.1, help="This parameter represents the probability of indicating a defect during inspection when, in fact, it does not exist.")
        beta=st.number_input("Insert the false-negative probability (\u03B5)", min_value = 0.0, max_value=1.0, value = 0.15, help="This parameter represents the probability of not indicating a defect during inspection when, in fact, it does exist.")
        ci=st.number_input("Insert cost of inspection (C_{I})", min_value = 0.0, value = 0.05, help="This parameter represents the cost of conducing an inspection.")
        cr=st.number_input("Insert cost of replacement (inspections and age-based) (C_{R})", min_value = 0.0, value = 1.0, help="This parameter represents the cost associated with preventive replacements, whether performed during inspections or when the age-based threshold is reached.")
        cf=st.number_input("Insert cost of failure (C_{F})", min_value = 0.0, value = 10.0, help="This parameter represents the replacement cost incurred when a component fails.")
        cd=st.number_input("Insert cost of defective by time unit (C_{D})", min_value = 0.0, value = 0.01, help="This parameter represents the unitary cost associated with the time in which the component stays in defective state for each time unit.")
        
        col1, col2 = st.columns(2)
        
        Delta=[0]
        st.subheader("Insert the variable values below:")
        K=int(st.text_input("Insert the number of inspections (K)", value=4))
        if (K<0):
            K=0
        Value=2
        if (K>0):
            for i, col in enumerate(st.columns(K)):
                col.write(f"**{i+1}-th inspection:**")
                Delta.append(col.number_input("Insp. Mom. (Δ)", value=Value*(i+1), min_value=Delta[i-1], key=f"Delta_{i}"))
        T = st.number_input("Insert the moment for the age-based preventive action (T)",value=(K+1)*Value,min_value=Delta[-1])
        
        st.subheader("Click on botton below to run this application:")    
        botao = st.button("Get cost-rate")
        if botao:
            st.write("---RESULT---")
            st.write("Cost-rate", KD_KT(K, Delta, T))
         
    if choice == menu[1]:
        st.header(menu[1])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>This app is dedicated to compute the cost-rate for a hybrid aperiodic inspection and age-based maintenance policy. We assume a single system operating under Delay-Time Modeling (DTM) with a heterogeneous component lifetime, each having distinct defect arrival distributions. Component renovation occurs either after a failure (corrective maintenance) or during inspections, once a defect is detected or if the age-based threshold is reached (preventive maintenance). We considered false-positive and false-negative probabilities during the inspection.</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>The app computes the cost-rate for a specific solution—defined by the number of inspections (K), inspection moments (Δ) and the age-based threshold (T).</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>For further questions or information on finding the optimal solution, please contact one of the email addresses below.</h6>", unsafe_allow_html=True)
        
        st.write('''

v.h.r.lima@random.org.br

r.g.n.paiva@random.org.br

a.j.s.rodrigues@random.org.br

h.s.j.gonzalez@random.org.br

c.a.v.cavalcante@random.org.br

''' .format(chr(948), chr(948), chr(948), chr(948), chr(948)))       
    if choice==menu[2]:
        st.header(menu[2])
        
        st.write('''The Research Group on Risk and Decision Analysis in Operations and Maintenance was created in 2012 
                 in order to bring together different researchers who work in the following areas: risk, maintenance a
                 nd operation modelling. Learn more about it through our website.''')
        st.markdown('[Click here to be redirected to our website](https://sites.ufpe.br/random/#page-top)',False)        
if st._is_running_with_streamlit:
    main()
else:
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
