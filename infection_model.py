#import statements
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

#xData = np.loadtxt ("F3Cblue.dat")

def model1(x,t):
    B=2.32*10**-4
    delta=38.1
    p=4.78*10**5
    c=0.117
    Rp = 1
    Rd = 1    
    #gamma = 1.02672 #SARS
    #k = 23.184
    #gamma = 1.656 #MERS
    #k = 125.424
    gamma = 0.8136 #MJHKU4r
    k = 133.488

    T= x[0]
    I = x[1]
    F1 = x[2]
    F2 = x[3]
    S = x[4]
    V = x[5]
    dTdt= -(B*T*V) - gamma*T*(I+S)
    dIdt= (B*T*V) - gamma*I*(T+2*I+S) - delta*I
    dF1dt= gamma*T*(2*I+S) + gamma*I*(2*I+S) - k*F1
    dF2dt= k*F1 - k*F2
    dSdt= k*F2 - Rd*delta*S
    dVdt= p*I + Rp*p*(S) - c*V
    return [dTdt,dIdt,dF1dt,dF2dt,dSdt,dVdt]

#params = [ 7.13*10**(-4),  50, 0.0161, 2] #SARS
#params = [ 1.15*10**(-3),  50, 0.0871, 2] #MERS
#params = [ 5.65*10**(-4),  50, 0.0927, 2] #MJHKU4r

t = np.linspace(0,60,600)
y0=[1, 0, 0, 0, 0, 0.01] 
print(y0)
ym = odeint(model1,y0, t)
data=np.column_stack([t,ym[:,:]])

np.savetxt("MJHKU4r_infection_noFp.dat",data)    
