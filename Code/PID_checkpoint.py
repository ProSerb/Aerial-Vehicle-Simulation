"""
Trifko Basic
PID Control trajectory tracking
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
from kinematics_3 import *

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 80
T = 0.04

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
    quadrant, sign = quadrant2(x[2])
    f = np.zeros(6) 
    f[0] = x[3]
    f[1] = x[4]
    f[2] = x[5]
    f[3] = ((u[0]*np.cos(np.pi/2 - x[2]) + u[1]*np.cos(np.pi/2 - x[2]) + u[2]*np.cos(np.pi/2 - x[2]))*1/m)*sign
    f[4] = (u[0]*np.sin(np.pi/2 - x[2]) + u[1]*np.sin(np.pi/2 - x[2]) + u[2]*np.sin(np.pi/2 - x[2]))*1/m 
    f[5] = (u[1]*np.sin(np.pi/2)-u[0]*np.sin(np.pi/2))*(l/I)
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

def quadrant(hyp, x1, x2):
    c = np.arccos((x2[0]-x1[0])/hyp)
    c = np.cos(c)
    s = np.arcsin((x2[1]-x1[1])/hyp)
    s = np.sin(s)
    if c > 0 and c < 1 and s > 0 and s < 1:
        quad = 'I'
    elif c < 0 and c > -1 and s > 0 and s < 1:
        quad = 'II'
    elif c < 0 and c > -1 and s < 0 and s > -1:
        quad = 'III'
    elif c > 0 and c < 1 and s < 0 and s > -1:
        quad = 'IV'    
    return quad

def quadrant2(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    if c > 0 and c < 1 and s > 0 and s < 1:
        quad = 'I'
        sign = -1
    elif c < 0 and c > -1 and s > 0 and s < 1:
        quad = 'II'
        sign = -1
    elif c < 0 and c > -1 and s < 0 and s > -1:
        quad = 'III'
        sign = 1  
    elif c > 0 and c < 1 and s < 0 and s > -1:
        quad = 'IV' 
        sign = 1   
    return quad, sign

def straight_on(quad, theta):
    if quad == 'I':
        angle = -(np.pi/2 - theta)
    elif quad == 'II':
        angle = np.pi/2 - theta 
    elif quad == 'III':
        angle =  (np.pi/2 + theta)
    elif quad == 'IV':
        angle = -(np.pi/2 + theta)
    return angle

def back_first(quad, theta):
    if quad == 'I':
        angle = -(np.pi/2 - theta)
    elif quad == 'II':
        angle = np.pi/2 - theta 
    elif quad == 'III':
        angle = np.pi/2 + theta
    elif quad == 'IV':
        angle = np.pi/2 - theta
    return angle

    
#%%
# Desired Location
x_d = np.zeros((6,N))
x_d[0,:] = 300
x_d[1,:] = 0

# Initialize x
u = np.zeros((3, N)) #FL, FR, FB
u[0, 0] = 5
u[1, 0] = 5
u[2, 0] = 0
x = np.zeros((6, N))
x_dot = np.zeros((6, N))


y = np.zeros((6, N))
x[0, 0] = 800
x[1, 0] = 10000
x[2, 0] = 0.01
x_dot[:, 0] = f(x[:,0], u[:,0])

# Get checkpoints
checkpoints = np.array([
    [x_d[0,0], x_d[0,0]*1.05, (x[0,0]+x_d[0,0])/2, x[0,0]],
    [x_d[1,0], x[1,0]*0.9, x[1,0]*2,x[1,0]]
])
plt.plot(checkpoints[0],checkpoints[1],'o',color = 'C1')

plt.show

# Initialize error for PID
err = np.zeros((3,N))
err_dot = np.zeros((3,N))
cum_err = np.zeros((3,N))
cum_err_dot = np.zeros((3,N))

# Select checkpoint
cp = checkpoints[:,0:-1]; cp_alt = np.zeros((2,np.size(cp[0,:])))

for i in range(0,np.size(cp[0,:])):
    if cp[0,i] < x[0,0]:
        cp_alt[:,i] = cp[:,i]
max = np.max(cp_alt[0,:])
index = np.where(cp_alt[0,:]==max)

x_d[0,0] = cp_alt[0, index]
x_d[1,0] = cp_alt[1, index]

# Further initalize error
theta = np.abs(np.arctan((x_d[1,0]-x[1,0])/(x_d[0,0]-x[0,0])))
hyp = np.sqrt((x_d[1,0]-x[1,0])**2 + (x_d[0,0]-x[0,0])**2)
x1 = np.array([x[0,0],x[1,0]]); x2 = np.array([x_d[0,0],x_d[1,0]])
quad = quadrant(hyp,x1,x2)
x_d[2,0] = straight_on(quad, theta)
err[:,0] = np.array([x[0,0] - x_d[0,0], x[1,0] - x_d[1,0], x[2,0] - x_d[2,0]]) 
err_dot[:,0] = np.array([x_dot[3, 0] - 0, x_dot[4, 0] - 0, x_dot[5, 0] - 0]) 

cum_err[:,0] += (err[:,0])*T  # starting error
cum_err_dot[:,0] += (err_dot[:,0])*T  # starting error

Kp = 0.6
Kd = 0.9
Ki = 0.1

#%% 
# Simulate
for k in range(1,N):    
    x[:, k] = rk_four(f, x[:, k - 1], u[:, k - 1], T)
    x_dot[:, k] = f(x[:,k-1], u[:,k-1])
    # Select checkpoint
    cp_alt = np.zeros((2,np.size(cp[0,:])))
    for i in range(0,np.size(cp[0,:])):
        if cp[0,i] < x[0,k]:
            cp_alt[:,i] = cp[:,i]
                
    max = np.max(cp_alt[0,:])
    index = np.where(cp_alt[0,:]==max)
    x_d[0,k] = cp_alt[0, index]
    x_d[1,k] = cp_alt[1, index]     

    # For determining desired orientation for vehicle 
    theta = np.abs(np.arctan((x_d[1,k-1]-x[1,k-1])/(x_d[0,k-1]-x[0,k-1])))
    hyp = np.sqrt((x_d[1,k-1]-x[1,k-1])**2 + (x_d[0,k-1]-x[0,k-1])**2)
    x1 = np.array([x[0,k-1],x[1,k-1]]); x2 = np.array([x_d[0,k-1],x_d[1,k-1]])
    quad = quadrant(hyp,x1,x2)
 
    x_d[2,k] = straight_on(quad, theta)

    #Compute error
    err[:,k] = np.array([x[0,k-1] - x_d[0,k-1], x[1,k-1] - x_d[1,k-1], x[2,k-1] - x_d[2,k-1]])
    err_dot[:, k] = np.array([x_dot[3, k-1] - 0, x_dot[4, k-1] - 0, x_dot[5, k-1] - 0])      
           
    cum_err[:,k] += (err[:,k])*T  # starting error
    de = (err[:,k] - err[:,k-1])/T 

    cum_err_dot[:,k] += (err_dot[:,k])*T  # starting error
    de_dot = (err_dot[:,k] - err_dot[:,k-1])/T 

    PID = Kp*err[2,k] + Ki*cum_err[2,k] + Kd*de[2]
    PID_dot = Kp*err_dot[1,k] + Ki*cum_err_dot[1,k] + Kd*de_dot[1]

    F = [2,2,10] #FL, FR, FB
    FR = F[1] - PID 
    FL = F[0] + PID
    if k < 3:
        FB = F[2]
    else:
        if err[1,k] > 0:
            FB = 0
        elif err[1, k] <= 0:
            FB = F[2]
    F = [FL,FR,FB] 
    u[:,k] = F

# %% 
# Plot Position
fig1 = plt.figure(figsize=(8, 6))
plt.plot(x[0,:],x[1,:], '--', color = "C0", label = 'Vehicle Trajectory')
plt.plot(x[0,0],x[1,0],'o',color = "C0", label = 'Starting Position')
plt.plot(cp[0,1:3],cp[1,1:3],'o',color = "C1", label = 'Checkpoint')
plt.plot(cp[0,0],cp[1,0],'*',color = "red", label = 'End point')
plt.ylabel(r"$y$ $[m]$")
plt.xlabel(r"$x$ $[m]$")
plt.legend()
plt.plot()

# Velocity and theta
fig2 = plt.figure(2)
fig2.set_figheight(5)
ax1a = plt.subplot(311)
plt.plot(t, x_dot[0,:],'--', color = "C1")
plt.grid()
plt.ylabel(r"$v_{x}$ $[m/s]$")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x_dot[1,:],'--', color = "C1")
plt.ylabel(r"$v_{y}$ $[m/s]$")
plt.grid()
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t,x[2,:], '--', color = "C1", label = 'Actual')
plt.plot(t,x_d[2,:], color = "C0", label = 'Desired')
plt.ylabel(r"$\theta$ $[rad]$")
plt.xlabel(r"$t$ $[s]$")
plt.grid()
plt.ylabel(r"$\theta$ $[rad]$")

plt.xlabel(r"$t$ [s]")
plt.legend()
plt.show()

# # Plot Forces
# fig2 = plt.figure()
# plt.plot(t, u[0,:], "C2", label = 'FL')
# plt.plot(t, u[1,:], "C1", label = "FR")
# plt.plot(t, u[2,:], "C3", label = "FB")
# plt.ylim(1.5,2.5)

# plt.ylabel(r"$Force$ $[N]$")
# plt.xlabel(r"$t$ $[s]$")
# plt.grid()
# plt.legend()
# plt.plot()




# %%
