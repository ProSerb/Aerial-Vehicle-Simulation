import numpy as np

def box_coor(center,l,w):
    TR = (center[0]+l,center[1]+w)
    TL = (center[0]-l,center[1]+w)
    BR = (center[0]+l,center[1]-w)
    BL = (center[0]-l,center[1]-w)
    return TR,TL,BR,BL

def rotate(x, y, center, theta):
    x_new = (x-center[0])*np.cos(theta)-(y-center[1])*np.sin(theta)
    y_new = (x-center[0])*np.sin(theta)+(y-center[1])*np.cos(theta)
    xy = [x_new, y_new]
    return xy

def gravi(t,y,g):
    f = np.zeros(2)
    f[0] = y[1]
    f[1] = -g
    return f

def model(t,y,g,I,F1,F2,m,l,w,psi,theta,sign1,sign2):
    f = np.zeros(6)    
    f[0] = y[3]
    f[1] = y[4]
    f[2] = y[5]
    f[3] = (F1*np.cos(psi - theta)+F2*np.cos(psi-theta))*(-(sign1))*(1/m)
    f[4] = (F1*np.sin(psi - theta)+F2*np.sin(psi-theta))*(1/m) - g
    f[5] = (F1*np.sin(psi-theta)*l + F1*np.cos(psi-theta)*w
    - F2*np.sin(psi-theta)*l - F2*np.cos(psi-theta)*w)*(1/I)
    return f

def desired_trajectory(t,y,F,w,l,g,I,m):
    f =np.zeros(6)
    psi = np.pi/2
    f[0] = y[3]
    f[1] = y[4]
    f[2] = y[5]
    f[3] = 0
    f[4] = (F[0]+F[1])*(1/m) - g
    f[5] = 0
    return f

def kin(t,y,g,I,F,m,l,w): #CLEAN UP 0 DEG CASE    
    f = np.zeros(6)   
    psi = np.pi/2 #90 deg in rads
    # if theta is positive
    if y[2] > 0:
        if np.cos(y[2]) == 1:
            theta = y[2]; sign1 = 1; sign2 = -1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
        # Quadrant I
        if np.cos(y[2])<1 and np.cos(y[2])>0 and np.sin(y[2])>0 and np.sin(y[2]) <1:
            theta = y[2]; sign1 = 1; sign2 = -1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        # Quadrant II
        if np.cos(y[2])>-1 and np.cos(y[2])<0 and np.sin(y[2])>0 and np.sin(y[2]) <1:
            theta = np.pi - y[2]; sign1 = 1; sign2 = 1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        # Quadrant III
        if np.cos(y[2])>-1 and np.cos(y[2])<0 and np.sin(y[2])<0 and np.sin(y[2]) >-1:
            theta = 3/2*(np.pi) - y[2]; sign1 = -1; sign2 = 1;
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        # Quadrant IV
        if np.cos(y[2])<1 and np.cos(y[2])>0 and np.sin(y[2])<0 and np.sin(y[2]) >-1:
            theta = 2*(np.pi) - y[2]; sign1 = -1; sign2 = -1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

    elif y[2] < 0: # theta is negative
        if np.cos(y[2]) == 1:
            theta = y[2]; sign1 = 1; sign2 = -1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
        #Quadrant I
        if np.cos(y[2])<1 and np.cos(y[2])>0 and np.sin(y[2])>0 and np.sin(y[2]) <1:
            theta = y[2] + 2*np.pi; sign1 = 1; sign2 = -1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        #Quadrant II
        if np.cos(y[2])>-1 and np.cos(y[2])<0 and np.sin(y[2])>0 and np.sin(y[2]) <1:
            theta = y[2] + (3/2)*np.pi; sign1 = 1; sign2 = 1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        #Quadrant III
        if np.cos(y[2])>-1 and np.cos(y[2])<0 and np.sin(y[2])>-1 and np.sin(y[2])<0:
            theta = y[2] + np.pi; sign1 = -1; sign2 = 1
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)

        #Quadrant IV
        if np.cos(y[2])<1 and np.cos(y[2])>0 and np.sin(y[2])>-1 and np.sin(y[2])<0:
            theta = y[2] + np.pi/2; sign1 = -1; sign2 = -1;
            f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
    elif y[2] == 0:
        theta = y[2]; sign1 = 1; sign2 = -1
        f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
    elif y[2] == np.pi/2:
        theta = y[2]; sign1 = 1; sign2 = 0
        f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
    elif y[2] == (3/2)*np.pi:
        theta = -y[2]; sign1 = -1; sign2 = 0
        f = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
    elif y[2] == np.pi:
        theta = y[2]; sign1 = 0; sign2 = -1
        f  = model(t,y,g,I,F[0],F[1],m,l,w,psi,theta,sign1,sign2)
 
    return f

def tracked_theta(theta):
    #Quadrant I
    if np.cos(theta)<1 and np.cos(theta)>0 and np.sin(theta)>0 and np.sin(theta) <1:
        n = np.ceil(theta/(np.pi/2))
        quadrant = 'I'
        theta_tracked = np.pi/2 - ((np.pi/2)*n - theta)
        sign1 = 1; sign2 = -1
    #Quadrant II
    elif np.cos(theta)>-1 and np.cos(theta)<0 and np.sin(theta)>0 and np.sin(theta) <1:
        n = np.ceil(theta/(np.pi))
        quadrant = 'II'
        theta_tracked = np.pi - ((np.pi)*n - theta)
        sign1 = 1; sign2 = 1
    #Quadrant III
    elif np.cos(theta)>-1 and np.cos(theta)<0 and np.sin(theta)>-1 and np.sin(theta)<0:
        n = np.ceil(theta/(np.pi*3/2))
        quadrant = 'III'
        theta_tracked = np.pi*(3/2) - ((np.pi*3/2)*n - theta)
        sign1 = -1; sign2 = 1;
    #Quadrant IV
    elif np.cos(theta)<1 and np.cos(theta)>0 and np.sin(theta)>-1 and np.sin(theta)<0:
        n = np.ceil(theta/(2*np.pi))
        quadrant = 'IV'
        theta_tracked = 2*np.pi - ((2*np.pi)*n - theta)
        sign1 = -1; sign2 = -1
    return quadrant, sign1, sign2, theta_tracked


   
