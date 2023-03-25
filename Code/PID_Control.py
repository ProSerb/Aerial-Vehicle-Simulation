"""
Trifko Basic
PID Control
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
SIM_TIME = 50
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
friction = -0.5

def f(x,u):
    quadrant, sign = quadrant2(x[2])
    f = np.zeros(6) 
    f[0] = x[3]
    f[1] = x[4]
    f[2] = x[5]
    f[3] = ((u[0]*np.cos(np.pi/2 - x[2]) + u[1]*np.cos(np.pi/2 - x[2]) + u[2]*np.cos(np.pi/2 - x[2]))*1/m)*sign
    f[4] = (u[0]*np.sin(np.pi/2 - x[2]) + u[1]*np.sin(np.pi/2 - x[2]) + u[2]*np.sin(np.pi/2 - x[2]))*1/m - g
    f[5] = (u[1]*np.sin(np.pi/2)-u[0]*np.sin(np.pi/2))*(l/I)
    return f

def contact(x,u):
    quadrant, sign = quadrant2(x[2])
    f = np.zeros(6) 
    f[0] += x[3]
    f[1] = 0
    f[2] = 0
    f[3] += x[3]*friction
    f[4] = 0
    f[5] = 0
    return f
def motionless(x,u):
    f = np.zeros(6) 
    return f

def box_coor(center,l):
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
    elif np.round(theta, 3) == 0:
        sign = 0
        quad = 'we good'
    return quad, sign

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
x_d[0,:] = 0
x_d[2,0] = 0
# Initialize x
u = np.zeros((3, N)) #FL, FR, FB
u[0, 0] = 5
u[1, 0] = 5
u[2, 0] = 0
x = np.zeros((6, N))
x_dot = np.zeros((6, N))

y = np.zeros((6, N))
x[0, 0] = 100
x[1, 0] = 10
x[2, 0] = 0
x_dot[:, 0] = f(x[:,0], u[:,0])

# Initialize error for PID
err = np.zeros((3,N))
cum_err = np.zeros((3,N))

# Further initalize error
err[:,0] = np.array([x[0,0] - x_d[0,0], x[1,0] - x_d[1,0], x[2,0] - x_d[2,0]]) 
cum_err[:,0] += (err[:,0])*T  # starting error

Kp = 0.6
Kd = 0.9
Ki = 0.1

#%% 
# Simulate
for k in range(1,N): 
    x_dot[:, k] = contact(x[:,k-1], u[:,k-1])
    if np.round(x[1, k-1]) > 0:
        x[:, k] = rk_four(f, x[:, k - 1], u[:, k - 1], T)        
    elif np.round(x[1,k-1]) == 0:
        #print("YOO")
        # if x_dot[3,k-1] >= g*friction:
            x[:, k] = rk_four(contact, x[:, k - 1], u[:, k - 1], T)
        # elif x_dot[3, k-1] <= g*friction:
        #     x[:, k] = rk_four(motionless, x[:, k - 1], u[:, k - 1], T)     

    F = [5,5,25] #FL, FR, FB
    if T*k <= 5: # Fly Straight up for 5 secs
        #Compute error
        x_d[2,k] = 0
        err[:,k] = np.array([x[0,k-1] - x_d[0,k-1], x[1,k-1] - x_d[1,k-1], x[2,k-1] - x[2,k-1]])      
        cum_err[:,k] += (err[:,k])*T  # starting error
        de = (err[:,k] - err[:,k-1])/T
        FR = F[1] 
        FL = F[0]
        FB = F[2]
    
    if T*k > 5 and T*k <= 15: #For another 5 seconds fly at an angle of 80 degrees
        #Compute error
        x_d[2,k] = 0.25
        err[:,k] = np.array([x[0,k-1] - x_d[0,k-1], x[1,k-1] - x_d[1,k-1], x[2,k-1] - x_d[2,k-1]])      
        cum_err[:,k] += (err[:,k])*T  # starting error
        de = (err[:,k] - err[:,k-1])/T 
        PID = Kp*err[2,k-1] + Ki*cum_err[2,k-1] + Kd*de[2]    
        FR = F[1] - PID 
        FL = F[0] + PID
        FB = F[2]
    
    if T*k > 15 and T*k <= 25: # Kill motors
        #Compute error
        x_d[2,k] = 0.25
        err[:,k] = np.array([x[0,k-1] - x_d[0,k-1], x[1,k-1] - x_d[1,k-1], x[2,k-1] - x_d[2,-1]])      
        cum_err[:,k] += (err[:,k])*T  # starting error
        de = (err[:,k] - err[:,k-1])/T 
        PID = Kp*err[2,k-1] + Ki*cum_err[2,k-1] + Kd*de[2]    
        FR = 0 
        FL = 0
        FB = 0     

    if T*k > 25: # Begin descent back to ground 
        #Compute error
        x_d[2,k] = 0
        err[:,k] = np.array([x[0,k-1] - x_d[0,k-1], x[1,k-1] - x_d[1,k-1], x[2,k-1] - x_d[2,-1]])      
        cum_err[:,k] += (err[:,k])*T  # starting error
        de = (err[:,k] - err[:,k-1])/T 
        PID = Kp*err[2,k-1] + Ki*cum_err[2,k-1] + Kd*de[2]    
        FR = F[1] - PID 
        FL = F[0] + PID
        FB = 10

    F = [FL,FR,FB] 
    u[:,k] = F

# %% 
# Make some plots
fig1 = plt.figure(figsize=(8, 6))
plt.plot(x[0,:],x[1,:],"C1", label = 'Vehicle Trajectory')
plt.plot(x[0,0],x[1,0],'o',color = "C1" , label = 'Starting Point')
plt.ylabel(r"$y$ $[m]$")
plt.xlabel(r"$x$ $[m]$")
plt.legend()
plt.show

fig2 = plt.figure(figsize=(8, 2))
plt.plot(t,x[2,:],'--', color = "C1", label = 'Actual')
plt.plot(t,x_d[2,:],color = "C0" , label = 'Desired')
plt.ylabel(r"$\theta$ $[rad]$")
plt.xlabel(r"$t$ $[s]$")
plt.legend()
plt.grid()
plt.show


# # Plot the states as a function of time
# fig3 = plt.figure(3)
# fig3.set_figheight(6.4)
# ax1a = plt.subplot(411)
# plt.plot(t, err[2,:], "C0")
# plt.grid()
# plt.ylabel(r"$\theta$ [m]")
# plt.setp(ax1a, xticklabels=[])
# ax1b = plt.subplot(412)
# plt.plot(t, x[2,:], "C0")
# plt.plot(t,x_d[2,:],"C1")
# plt.grid()
# plt.ylabel(r"$\theta$ [m]")
# plt.setp(ax1b, xticklabels=[])
# ax1c = plt.subplot(413)
# plt.plot(t, u[0,:], "C1")
# plt.plot(t, u[1,:],"C2")
# #plt.plot(t, u[2,:],"C3")
# plt.grid()
# plt.ylabel(r"$Force$ [N]")
# plt.setp(ax1c, xticklabels=[])
# ax1d = plt.subplot(414)
# plt.plot(t, err[0,:], "C0")
# plt.grid()
# plt.ylabel(r"$x$ [m]")
# plt.setp(ax1c, xticklabels=[])
# plt.xlabel(r"$t$ [s]")

# fig4 = plt.figure(figsize=(8, 6))
# plt.plot(t,x[3,:],"C0")

# plt.legend()
# plt.grid()

# %%
# Set up vehicle for animation
center = [x[0,:],x[1,:]]
coors = box_coor(center,l)
TR = coors[0]
TL = coors[1]
BR = coors[2]
BL = coors[3]

# Apply tilt to box corners
TR = rotate(TR[0], TR[1], center, x[2,:])
TL = rotate(TL[0], TL[1], center, x[2,:])
BR = rotate(BR[0], BR[1], center, x[2,:])
BL = rotate(BL[0], BL[1], center, x[2,:])

# Add center position bax to box corners
TR = (TR[0]+center[0], TR[1]+center[1])
TL = (TL[0]+center[0], TL[1]+center[1])
BR = (BR[0]+center[0], BR[1]+center[1])
BL = (BL[0]+center[0], BL[1]+center[1])

# Force Vectors
FL = [TL[0], TL[1] - l]
FL = rotate(FL[0], FL[1],center,x[2,:]) 
FL = (FL[0]+center[0],FL[1]+center[1])

FR = [TR[0], TR[1] - l]
FR = rotate(FR[0], FR[1],center,x[2,:]) 
FR = (FR[0]+center[0],FR[1]+center[1])

FB = [BL[0] + l, BL[1]]
FB = rotate(FB[0], FB[1],center,x[2,:]) 
FB = (FB[0]+center[0],FB[1]+center[1])

# plt.plot(TL[0][0],TL[1][0],'o')
# plt.plot(TR[0][0],TR[1][0],'o')
# plt.plot(FL[0][0],FL[1][0],'o')
# plt.plot(BL[0][0],BL[1][0],'o')
# plt.plot(FB[0][0],FB[1][0],'o')

dxL = FL[0] - BL[0]
dyL = FL[1] - BL[1]
dxR = FR[0] - BR[0]
dyR = FR[1] - BR[1]
dxB = center[0] - FB[0]
dyB = center[1] - FB[1]

# p.arrow(BL[0][0],BL[1][0]+l,-dxL[0]*1.5,-dyL[0]*1.5,color='green',head_width = 0.1)
# p.arrow(FB[0][0],FB[1][0],-dxB[0],-dyB[0],color='green',head_width = 0.1)
# p.arrow(BR[0][0],BR[1][0]+l,-dxR[0]*1.5,-dyR[0]*1.5,color='green',head_width = 0.1)

# plt.show

# %%
# Animate
fig = plt.figure(figsize=(8, 6))
d = 20
ax = plt.axes(xlim=(x[0,0]-d,x[0,0]+d),ylim=(x[1,0]-d,x[1,0]+d))

box, = ax.plot([],[],'-',lw=2)
#FL_arrow, = p.arrow([],[],[],[],color='orange',head_width = 0.5)
x2 = []; y2 = []
xg = []; yg = []

ground = np.array([
    [np.min(x[0,:])-d,np.max(x[0,:])+d],
    [0 - l, 0 - l]
])

def animate(i):
    x1 = [TL[0][i], TR[0][i], BR[0][i], BL[0][i], TL[0][i]]
    y1 = [TL[1][i], TR[1][i], BR[1][i], BL[1][i], TL[1][i]]
    x2.append(x[0,i])
    y2.append(x[1,i])
    box.set_data([x1],[y1])
    ax.set_ylim(bottom = x[1,i]-d, top = x[1,i]+d)
    ax.set_xlim(left = x[0,i]-d, right = x[0,i]+d)
    ax.plot(x2,y2,'--',color="gray")
    ax.plot(ground[0],ground[1],'-',color="black")
    

    #FL_arrow.set_data([BL[0][i]],[BL[1][i]+l],[-dxL[i]*1.5],[-dyL[i]*1.5])

    #p.arrow(xL,yL,-dxL[i]*1.5,-dyL[i]*1.5,color='orange',head_width = 0.5)
    #p.arrow(FB[0][i],FB[1][i],-dxB[i],-dyB[i],color='red',head_width = 0.5)
    #p.arrow(BR[0][i],BR[1][i]+l,-dxR[i]*1.5,-dyR[i]*1.5,color='orange',head_width = 0.5)
    return box,

anim = animation.FuncAnimation(fig, animate, frames=100, blit = True)

HTML(anim.to_jshtml())

# %%
