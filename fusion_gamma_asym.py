#import statements
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

#xData = np.loadtxt ("F3Cblue.dat")

def virus(x,t, g, k, n):
    dt = np.zeros(n+3)
    D = x[0]
    A = x[1]
    S = x[-1]
    dt[0] = -g*D*A 
    dt[1] = (-g*D*A) - (g*S*A)
    dt[2] = (2*g*D*A) + (g*S*A) - k*x[2]
    for i in range(3,n+2):
        dt[i] = k*x[i-1]-k*x[i]
    dt[n+2] = k*x[n+1]
    return dt

#params = [ 7.13*10**(-4),  50, 0.0161, 2] #SARS
#params = [ 1.15*10**(-3),  50, 0.871, 2] #MERS
params = [ 5.65*10**(-4),  50, 0.0927, 2] #MJHKU4r

t = np.linspace(0,240,100)
y0=np.zeros(params[3]+3)
y0[0] = params[1] 
y0[1] = 100-params[1]
print(y0)
ym = odeint(virus,y0, t, args = (params[0], params[2], params[3] ))
data=np.column_stack([t,ym[:,:]])

np.savetxt("MJHKU.dat",data)    
