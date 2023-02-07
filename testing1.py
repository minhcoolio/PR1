#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


'''  
test = imu_vals[:, :1]
print(test)
x = np.average(test, axis=1)
print(x)
test1 = test-x
print(test1)
'''
#print(qt_exp(q0))

'''
at time t we have image and orientation
project image to a sphere fov 60deg hori and 45deg vert 0,0 deg at center
    2 angle w/ every pixel
down is positive (rotate around y) RHR
find angles for each pixel
put pixels at certain radius on sphere on sphere in body frame
need to transform to world frame
rotate by orientaiton to world frame
go from sphereical to cartesian
    try stacking all pixels and multiplying it by the Rotation matrix (makes it faster) left and down is pos (find func from spherical to cart with this convention)
    multiply by R
now with pixel in world frame place on cylinder, cut and unwrap to get panorama

grad descent
we are doing the optimization not learning the optimization
'''
q1 = np.array([1,2,3,4])
q2 = np.array([20,-2,100,53])
x = qt_multiply(q1, q2)
print(x)
y = qt_exp(q2)
print(y)

print(vic_mats[:,:,2500])
print(vic_mats[:,:,0])
dif1 = vic_mats[:,:,0]-vic_mats[:,:,900]
dif2 = vic_mats[:,:,0]-vic_mats[:,:,2000] #1622
print(LA.norm(dif1))
print(LA.norm(dif2))

print(q1-q2)

'''
print("Testing")
print(np.amax(imu_phys[3, :]))
print(np.amax(imu_phys[4, :]))
print(np.amax(imu_phys[5, :]))
'''


#print(imu_vals[0:3,1])
#print(imu_vals[0:3,10])
#for i in range(100):
#    print(vic_vals[:,:,i]) #take average of bias values of first few time steps
'''
print(vic_ts[0,1]-vic_ts[0,0])
print(vic_ts[0,101]-vic_ts[0,100])

q_test = np.array([1,0,0,0])
print(q_test[0])

q_test = np.array([1,2,3,4])
print(qt_exp(q_test))
x = np.append(20, q_test)
print(x)

q_test1 = np.array([2,2,2,2])
q_test2 = np.array([1,.5,.5,.75])
out = qt_multiply(q_test1, q_test2)
print(out)
'''

'''
ABC################
q_nxt = np.zeros([4, N])
q_nxt[:, 0] = q0
ang = np.zeros([3,N])
for j in range(1,N):
    tau = imu_ts[0,j]-imu_ts[0,j-1]
    w = np.array([ imu_phys[4,j], imu_phys[5,j], imu_phys[3,j] ]) # order from imu is zxy but we need xyz
    q_nxt[:, j] = motion_mod(q_nxt[:, j-1], w, tau)
    ang[:, j] = t3d.euler.quat2euler(q_nxt[:, j], 'rzyx')
'''

"""

import pickle
import sys
import time 
#import numpy as np
import transforms3d as t3d

from autograd import grad
import autograd.numpy as np
'''
import jax.numpy as np
from jax import grad, jit
from jax.numpy import linalg as LA
'''
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d as a3
#from rotplot import rotplot

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

def find_avg_bias(data): #time_lim is time point to avg all pts under
    avg = np.average(data, axis=1)
    return avg

#def find_lim():
    

def calibrate(raw): #pass in raw vector (not matrix)
    time_lim = 1500 # use vicon to figure out how long it isnt moving for, fix this
    N = raw.shape[1]
    value = np.zeros(raw.shape)
    
    # find average of first couple time steps
    biasW = find_avg_bias(raw[0:3, :time_lim])
    biasA = find_avg_bias(raw[3:6, :time_lim])
    
    for i in range(N):
        Vref = 3300 #in mV, 3.3V
        senW = 3.33 * 180/np.pi # sensitivity mV/rad/s
        scale_factorW = Vref/1023/senW
        
        Vref = 3300 #in mV, 3.3V
        senA = 3 # sensitivity mV/g
        scale_factorA = Vref/1023/senA
        

        noiseA = np.array([140.162, 126.888, 228.638])
        noiseW = np.array([-138.883333, -126.460666, -228.994])
        '''
        if i < 100:
            #print(raw[3:6, i] - biasW)
            print(raw[0:3, i] - biasA)
        '''
        #value[0:3, i] = (raw[0:3, i] - biasA - noiseA) * scale_factorA + np.array([0,0,1])
        #value[3:6, i] = (raw[3:6, i] - biasW - noiseW) * scale_factorW 
        value = value.at[0:3, i].set((raw[0:3, i] - biasA - noiseA) * scale_factorA + np.array([0,0,1]))
        value = value.at[3:6, i].set((raw[3:6, i] - biasW - noiseW) * scale_factorW)
        
    return value

# multiply 2 quaternions which are (4,) np arrays 
def qt_multiply(q1, q2):
    q1v = np.array([q1[1], q1[2], q1[3]]) #removing the scalar factor to get just qv
    q2v = np.array([q2[1], q2[2], q2[3]]) #removing the scalar factor to get just qv
    prod1 = np.array([ q1[0]*q2[0]-np.dot(q1v, q2v)])
    prod2 = q1[0]*q2v + q2[0]*q1v + np.cross(q1v, q2v)
    prod = np.append(prod1, prod2)
    return prod
    

# take exponential of a quat
# q is a quat. a (4,) np array
def qt_exp(q): 
        ex = np.exp(q[0])
        qv = np.array([q[1], q[2], q[3]]) #removing the scalar factor to get just qv
        sinQs = np.sin(LA.norm(qv))/LA.norm(qv) #error occuring here where qv is most likely 0
        expq = ex*np.array([np.cos(LA.norm(qv)), q[1]*sinQs, q[2]*sinQs, q[3]*sinQs])
        return expq
    
def qt_inv(q):
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    q_inv = q_conj/(LA.norm(q)**2)
    return q_inv
    
def qt_log(q):
    qv = np.array([q[1], q[2], q[3]]) #removing the scalar factor to get just qv
    temp1 = np.log(LA.norm(q))
    acos = np.arccos(q[0]/LA.norm(q))
    temp2 = qv*acos/LA.norm(qv)
    qtlog = np.append(temp1, temp2)
    
    if q[1] == 0 and q[2] == 0 and q[3] == 0:
        qtlog = np.array([np.log(LA.norm(q[0])), q[1], q[2], q[3]])
    return qtlog

# observation model
def h(q):
    mid = np.array([0,0,0,-1])
    temp1 = qt_multiply(qt_inv(q), mid)
    out = qt_multiply(temp1, q)
    return out

# computes motion model for next time step
# w is ang vel and q is a quat a (3,) array
# tau is time interval between ang. vel. data pts, it is one number
def motion_mod(q, w, tau):
    temp1 = tau*w/2
    temp2 = np.append(0, temp1)
    exp1 = qt_exp(temp2)
    
    Fqw = qt_multiply(q, exp1)
    return Fqw
    
# computes cost function
# input q_traj is entire quaternion trajectory (4,T) array, ts is the timestamps
    
#@jit
def cost_fn(q_tj, imu_data, ts):
    T = q_tj.shape[1]
    at = imu_data[:3, :] #imu lin accel
    mm_err = np.array([]) # motion model error
    ob_err = np.array([]) # observation model error
    for i in range(T-1):
        tau = ts[0, i+1]-ts[0, i]
        w = np.array([imu_data[4,i], imu_data[5,i], imu_data[3,i]])
        temp1 = qt_multiply( qt_inv(q_tj[:, i+1]), motion_mod(q_tj[:,i], w, tau) )
        temp2 = LA.norm(2*qt_log(temp1))**2
        mm_err = np.append(mm_err, temp2) #collection of all mm error terms
        
    for j in range(T):
        temp3 = np.append(0, at[:3, j]) #need to append a zero to make it (4,)
        temp4 = LA.norm(temp3 - h(q_tj[:,j]))**2
        ob_err = np.append(ob_err, temp4)
    
    out = .5*np.sum(mm_err) + .5*np.sum(ob_err)
    return out


dataset="1"
cfile = "../data/cam/cam" + dataset + ".p"
ifile = "../data/imu/imuRaw" + dataset + ".p"
vfile = "../data/vicon/viconRot" + dataset + ".p"

ts = tic()
#camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")


###################
imu_vals = imud.get("vals")
imu_ts = imud.get("ts")

vic_mats = vicd.get("rots")
vic_ts = vicd.get("ts")
N = vic_ts.shape[1] # total # of data pts
###################
# Taking raw values from above and turning 
# them into physical units

#print(imu_vals)
#print(imu_vals[0:3,1]) #first 3 in column
#print(imu_vals[3:6,1]) #last 3 in column
#print(imu_vals[:,0])

imu_phys = calibrate(imu_vals)

q0 = np.array([1, 0, 0, 0])

# convert all VICON data to euler angles
vic_y = [0]*N
vic_p = [0]*N
vic_r = [0]*N
for i in range(N):
    vic_y[i], vic_p[i], vic_r[i] = t3d.euler.mat2euler(vic_mats[:,:,i], 'rzyx')


# create motion model for quats and convert them to rotation matrices #ABC
q_nxt = np.zeros([4, N])
#q_nxt[:, 0] = q0
q_nxt = q_nxt.at[:, 0].set(q0)
ang = np.zeros([3,N])
for j in range(N-1):
    tau = imu_ts[0,j+1]-imu_ts[0,j]
    w = np.array([ imu_phys[4,j], imu_phys[5,j], imu_phys[3,j] ]) # order from imu is zxy but we need xyz
    #q_nxt[:, j+1] = motion_mod(q_nxt[:, j], w, tau) # quat trajectory
    q_nxt = q_nxt.at[:, j+1].set(motion_mod(q_nxt[:, j], w, tau))
    #ang[:, j] = t3d.euler.quat2euler(q_nxt[:, j], 'rzyx')
    ang = ang.at[:,j].set(t3d.euler.quat2euler(q_nxt[:, j], 'rzyx'))
    
mm_y = ang[0, :] #Split MM angles up into roll pitch yaw
mm_p = ang[1, :]
mm_r = ang[2, :]

# Plot all angles in radians
fig, axes = plt.subplots(3)
axes[0].set_title('Yaw in Radians')
axes[0].plot(range(N), vic_y, color='r', label='VICON')
axes[0].plot(range(N), mm_y, color='b',label='MM')
axes[1].set_title('Pitch in Radians')
axes[1].plot(range(N), vic_p, color='r', label='VICON')
axes[1].plot(range(N), mm_p, color='b',label='MM')
axes[2].set_title('Roll in Radians')
axes[2].plot(range(N), vic_r, color='r', label='VICON')
axes[2].plot(range(N), mm_r, color='b',label='MM')
plt.show()

'''
print(q_nxt.shape[1])
temp1 = np.array([])
print(temp1)
temp1 = np.append(temp1, 1)
print(temp1)
temp1 = np.append(temp1, 2)
print(temp1.shape)
print(np.append(0, imu_phys[:3, 1]))
'''

cost_grad = grad(cost_fn)
gradient1 = cost_grad(q_nxt, imu_phys, imu_ts)
print(gradient1)

'''
print(q_nxt.shape[1])
temp1 = np.array([])
print(temp1)
temp1 = np.append(temp1, 1)
print(temp1)
temp1 = np.append(temp1, 2)
print(temp1.shape)
print(np.append(0, imu_phys[:3, 1]))
'''