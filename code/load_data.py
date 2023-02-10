import pickle
import sys
import time 
#import numpy as np
import transforms3d as t3d

from autograd import grad, jacobian
import autograd.numpy as np
from autograd.numpy import linalg as LA
'''
import jax.numpy as np
from jax import grad, jit
from numpy import linalg as LA
'''
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d as a3
#from rotplot import rotplot
from autograd.numpy import save


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
        senA = 300 # sensitivity mV/g
        scale_factorA = Vref/1023/senA
        
        #noiseA = np.array([140.162, 126.888, 228.638])
        noiseA = np.array([140.162, 126.888, 1/scale_factorA + 228.638])
        noiseW = np.array([-138.883333, -126.460666, -228.994])
 
        #print(raw[0:3, i] - biasA)
        #value[0:3, i] = (raw[0:3, i] - biasA - noiseA) * scale_factorA + np.array([0,0,1])
        value[0:3, i] = (raw[0:3, i] - biasA - noiseA) * scale_factorA
        value[3:6, i] = (raw[3:6, i] - biasW - noiseW) * scale_factorW 
        
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
        
        expq += 1e-6 #adding perturbation to avoid singularities
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
    
    tol = .01 #check for if qv = 0
    if LA.norm(qv) < tol and LA.norm(qv) > -tol:
        qtlog = np.array([np.log(np.absolute(q[0])), 0, 0, 0])
        
    qtlog += 1e-6 #adding perturbation to avoid singularities
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
        
    for j in range(1,T):
        temp3 = h(q_tj[:,j])
        qv = temp3[1:]
        temp4 = LA.norm(at[:3,j] - qv)**2
        ob_err = np.append(ob_err, temp4)
        '''
        temp3 = np.append(0, at[:3, j]) #need to append a zero to make it (4,)
        temp4 = LA.norm(temp3 - h(q_tj[:,j]))**2
        ob_err = np.append(ob_err, temp4)
        '''
    
    #print(np.sum(mm_err))
    #print(np.sum(ob_err))
    out = .5*np.sum(mm_err) + .5*np.sum(ob_err) #perturbing slightly
    return out


dataset="1"
cfile = "../data/cam/cam" + dataset + ".p"
ifile = "../data/imu/imuRaw" + dataset + ".p"
vfile = "../data/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")


###################
imu_vals = imud.get("vals")
imu_ts = imud.get("ts")

vic_mats = vicd.get("rots")
vic_ts = vicd.get("ts")
N = vic_ts.shape[1] # total # of data pts

cam_im = camd.get("cam")
cam_ts = camd.get("ts")
###################
# Taking raw values from above and turning 
# them into physical units
imu_phys = calibrate(imu_vals)

q0 = np.array([1.0, 0.0, 0.0, 0.0])

# convert all VICON data to euler angles
vic_y = [0]*N
vic_p = [0]*N
vic_r = [0]*N
for i in range(N):
    vic_y[i], vic_p[i], vic_r[i] = t3d.euler.mat2euler(vic_mats[:,:,i], 'rzyx')


# create motion model for quats and convert them to rotation matrices #ABC
# q_nxt will act as the initial step for the cost function
q_nxt = np.zeros([4, N])
q_nxt[:, 0] = q0
ang = np.zeros([3,N])
for j in range(N-1):
    tau = imu_ts[0,j+1]-imu_ts[0,j]
    w = np.array([ imu_phys[4,j], imu_phys[5,j], imu_phys[3,j] ]) # order from imu is zxy but we need xyz
    q_nxt[:, j+1] = motion_mod(q_nxt[:, j], w, tau) # quat trajectory
    ang[:, j] = t3d.euler.quat2euler(q_nxt[:, j], 'rzyx')
    
mm_y = ang[0, :] #Split MM angles up into roll pitch yaw
mm_p = ang[1, :]
mm_r = ang[2, :]

# Plot all angles in radians
fig, axes = plt.subplots(3)
fig.subplots_adjust(hspace=1.0)
axes[0].set_title('Yaw in Radians', pad=-10)
axes[0].plot(range(N), vic_y, color='r', label='VICON')
axes[0].plot(range(N), mm_y, color='b',label='MM')
axes[1].set_title('Pitch in Radians', pad=-10)
axes[1].plot(range(N), vic_p, color='r', label='VICON')
axes[1].plot(range(N), mm_p, color='b',label='MM')
axes[2].set_title('Roll in Radians', pad=-10)
axes[2].plot(range(N), vic_r, color='r', label='VICON')
axes[2].plot(range(N), mm_r, color='b',label='MM')
plt.show()

# Plot the accel in units of 1
#lim = imu_vals.shape[1]
lim = N
a_nxt = np.zeros([4,lim])
for z in range(lim):
    a_nxt[:, z] = h(q_nxt[:, z])
a_x = a_nxt[1, :]
a_y = a_nxt[2, :]
a_z = a_nxt[3, :]
fig2, axes = plt.subplots(3)
fig2.subplots_adjust(hspace=1.0)
axes[0].set_title('Ax')
axes[0].plot(range(N), imu_phys[0,:5561], color='r', label='VICON')
axes[0].plot(range(N), a_x, color='b',label='MM')
axes[1].set_title('Ay')
axes[1].plot(range(N), imu_phys[1,:5561], color='r', label='VICON')
axes[1].plot(range(N), a_y, color='b',label='MM')
axes[2].set_title('Az')
axes[2].plot(range(N), imu_phys[2,:5561], color='r', label='VICON')
axes[2].plot(range(N), a_z, color='b',label='MM')
plt.show()

    


'''
#testing the gradient cost function
####################################
cost_grad = grad(cost_fn)
gradient1 = cost_grad(q_nxt, imu_phys, imu_ts)
####################################
#print("END")
#print(gradient1[:, 1])
'''
#log_grad = jacobian(qt_log)
#testlog = log_grad(q0)
#print(testlog)

#log_grad = jacobian(qt_log)
#testlog = log_grad(q0)
#print(testlog)
#print(qt_log(q0))
T = imu_phys.shape[1]
q_nxt = np.zeros([4, T])
q_nxt[:, 0] = q0
ang = np.zeros([3,T])
for j in range(T-1):
    tau = imu_ts[0,j+1]-imu_ts[0,j]
    w = np.array([ imu_phys[4,j], imu_phys[5,j], imu_phys[3,j] ]) # order from imu is zxy but we need xyz
    q_nxt[:, j+1] = motion_mod(q_nxt[:, j], w, tau) # quat trajectory

# take q_nxt as t=0
num_iter = 0 # track number of iterations
q_curr = q_nxt/LA.norm(q_nxt, axis=0)
tol = .5
bail_count = 0
cost_grad = grad(cost_fn)
while num_iter < 5:
    
    start = time.time() #for testing
    gradi = cost_grad(q_curr, imu_phys, imu_ts)
    
    step = .5 # implement adaptive step size based on lipschitz
    q_nxt = q_curr - step*gradi      
    q_nxt /= LA.norm(q_nxt, axis=0) #norm each quaternion 
    num_iter = num_iter+1  
    
    test = np.absolute((LA.norm(q_curr-q_nxt))) #early bail out method
    if test < tol: 
        bail_count = bail_count + 1
        if bail_count == 10:
            break
    q_curr = q_nxt
    
    stop = time.time()
    duration = stop-start
    #print(num_iter)
    #print(duration)

# Saving optimized trajectory to avoid rerunning just for the same matrix
#save('optim_traj.npy', q_nxt)

print("Optimized Orientation")
ang_opt = np.zeros([3,N])
for j in range(N-1): # convert optimized quat traj to euler
    ang_opt[:, j] = t3d.euler.quat2euler(q_nxt[:, j], 'rzyx')
    
opt_y = ang_opt[0, :] #Split MM angles up into roll pitch yaw
opt_p = ang_opt[1, :]
opt_r = ang_opt[2, :]

# Plot all angles in radians
fig, axes = plt.subplots(3)
fig.subplots_adjust(hspace=1.0)
axes[0].set_title('Yaw in Radians')
axes[0].plot(range(N), vic_y, color='r', label='VICON')
axes[0].plot(range(N), opt_y, color='b',label='Optim')
axes[1].set_title('Pitch in Radians')
axes[1].plot(range(N), vic_p, color='r', label='VICON')
axes[1].plot(range(N), opt_p, color='b',label='Optim')
axes[2].set_title('Roll in Radians')
axes[2].plot(range(N), vic_r, color='r', label='VICON')
axes[2].plot(range(N), opt_r, color='b',label='Optim')
plt.show()