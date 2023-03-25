"""
Trifko Basic
Approx Linearization Attempt
4/23/2022
"""
# %%
from IPython.display import HTML
import numpy as np
from numpy import linalg
import scipy.integrate as int
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control
import pylab as p
from mobotpy.integration import rk_four

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30
T = 0.01

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

#%% 
# Setup Vehicle Parameters
L = 2
l = L/2
m = 2.5
g = 9.81
I = (L**4)/12

def f(x,u):
    f = np.zeros(6) 
    f[0] = x[3]
    f[1] = x[4]
    f[2] = x[5]
    f[3] = (u[0]*np.cos(np.pi/2 - x[2]) - u[1]*np.cos(np.pi/2 - x[2]) - u[2]*np.cos(np.pi/2 - x[2]))*1/m
    f[4] = (u[0]*np.sin(np.pi/2 - x[2]) + u[1]*np.sin(np.pi/2 - x[2]) + u[2]*np.sin(np.pi/2 - x[2]))*1/m - g 
    f[5] = (u[1]*np.sin(np.pi/2)-u[0]*np.sin(np.pi/2))*(l/I)
    #print(f[3],f[4],f[5])
    return f

def theta(x,u):
    f = x[0]
    return f

def box_coor(center,l,w):
    TR = (center[0]+l,center[1]+l)
    TL = (center[0]-l,center[1]+l)
    BR = (center[0]+l,center[1]-l)
    BL = (center[0]-l,center[1]-l)
    return TR,TL,BR,BL

def rotate(x, y, center, theta):
    x_new = (x-center[0])*np.cos(theta)-(y-center[1])*np.sin(theta)
    y_new = (x-center[0])*np.sin(theta)+(y-center[1])*np.cos(theta)
    xy = [x_new, y_new]
    return xy
    
#%%
# Desired trajectory
x_d = np.zeros((6,N))
x_d[0, 0] = 300
x_d[1, 0] = 300
x_d[2, 0] = -np.pi/4
u_d = np.zeros((3,N))
v_d = -0.5 # [m/s]

for k in range(0, N):
    u_d[:, k] = np.array([5,5,40])

for k in range (1,N):
    x_d[:,k] = rk_four(f, x_d[:, k - 1], u_d[:, k - 1], T)

# Initialize some x
u = np.zeros((3, N)) #FL, FR, FB
u[0, 0] = 5
u[1, 0] = 5
u[2, 0] = 40
x = np.zeros((6, N))
y = np.zeros((6, N))
x[0, 0] = 300
x[1, 0] = 500


#%%
P = np.zeros((3,N))
# Simulate
for k in range(1,N):    
    x[:, k] = rk_four(f, x[:, k - 1], u[:, k - 1], T)

    LR_ang =np.pi/2 - x_d[2, k -1]
    B_ang =  np.pi/2 - x_d[2, k -1]
    
    A = np.array([
        [0, 0, (-u_d[0,k-1]*np.sin(LR_ang) + u_d[1,k-1]*np.sin(LR_ang) + u_d[2,k-1]*np.sin(B_ang))*1/m],
        [0, 0, (u_d[0,k-1]*np.cos(LR_ang) + u_d[1,k-1]*np.cos(LR_ang) + u_d[2,k-1]*np.cos(B_ang))*1/m - g],
        [0, 0, (u_d[1,k-1]*np.cos(np.pi/4)-u_d[0,k-1]*np.cos(np.pi/4))*(l/I)]

    ])
    B = np.array(
        [
            [np.cos(LR_ang)*1/m, -np.cos(LR_ang)*1/m, -np.cos(B_ang)*1/m],
            [np.sin(LR_ang)*1/m, np.sin(LR_ang)*1/m, np.sin(B_ang)*1/m],
            [-np.sin(np.pi/4)*1/I, np.sin(np.pi/4)*1/I, 0],
        ]
    )
    # Compute the gain matrix to place poles of (A-BK) at p
    p = np.array([-0.25, -0.5, -0.85])
    K = signal.place_poles(A, B, p)
    P[:,k] = -K.gain_matrix @ (x[0:3, k - 1] - x_d[0:3, k - 1])
    u[:,k] = u_d[:,k-1] + P[:,k]


#%% 
fig1 = plt.figure(figsize=(8, 6))
plt.plot(x_d[0,:],x_d[1,:],"C0",label = 'Desired')
plt.plot(x[0,:],x[1,:],'--',color = "C1", label = 'Actual')
plt.plot(x[0,0],x[1,0],'o',color = "C1", label = 'Actual starting point')
plt.plot(x_d[0,0],x_d[1,0],'o',color = "C0", label = 'Desired starting point')
plt.ylabel(r"$y$ $[m]$")
plt.xlabel(r"$x$ $[m]$")
plt.legend()

# Plot the states as a function of time
fig2 = plt.figure(2)
fig2.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x[0,:],'--', color = "C1", label = 'Actual')
plt.plot(t, x_d[0,:], "C0", label = "Desired")
plt.grid()
plt.legend()
plt.ylabel(r"$x$ $[m]$")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(412)
plt.plot(t, x[1,:],'--', color = "C1", label = 'Actual')
plt.plot(t, x_d[1,:], "C0", label = "Desired")
plt.grid()
plt.legend()
plt.ylabel(r"$y$ $[m]$")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x[2,:],'--', color = "C1", label = 'Actual')
plt.plot(t, x_d[2,:], "C0", label = "Desired")
plt.grid()
plt.legend()
plt.ylabel(r"$\theta$ $[rad]$")
plt.setp(ax1c, xticklabels=[])

ax1d = plt.subplot(414)
plt.plot(t, u[0,:], "C2", label = 'FL')
plt.plot(t, u[1,:], "C1", label = "FR")
plt.plot(t, u[2,:], "C3", label = "FB")
plt.grid()
plt.ylabel(r"$Force$ $[N]$")
plt.setp(ax1c, xticklabels=[])
plt.xlabel(r"$t$ [s]")
plt.legend()
plt.grid()

# Plot the states as a function of time
fig3 = plt.figure(3)
ax1a = plt.subplot(211)
plt.plot(t, x[3,:],'--', color = "C1")
plt.grid()
plt.ylabel(r"$v_{x}$ $[m/s]$")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(212)
plt.plot(t, x[4,:],'--', color = "C1")
plt.ylabel(r"$v_{y}$ $[m/s]$")
plt.grid()
plt.xlabel(r"$t$ [s]")





# %%
