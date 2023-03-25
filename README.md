## Aerial-Vehicle-Simulation
#### Description:
Created a vehicle simulation in python to simulate a simplified model of an aerial vehicle with three thrusters to test out two control methods: approximate linearization control and PID control, to analyze the performance of the vehicle autonomously orientating itself and following a user-set desired trajectory.

_Program Language_: Python

#### Approx_linear.py:
Approximate Linearization control was used to have the vehicle follow a desired trajectory. This is a useful application because aerial vehicles 
example with the Falcon 9. It could be desirable to have the rocket on reentry follow a precomputed 
trajectory that is deem safest based on perhaps the weather, and maybe the wind traffic

#### PID_Control.py:
PID control was used as an alternative to approximate linearization. The main purpose of the PID control was to control the orientation of the vehicle. 

#### PID_checkpoint.py
To attempt to have the PID controller used and follow a desired trajectory it was assumed that the vehicle knows where its final destination
is and then computes a polynomial trajectory and places some checkpoints along that path for it to follow. Essentially the checkpoints are used as guide for the vehicle.

