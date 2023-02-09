# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:55:29 2023

@author: mante
"""

import pickle
import sys
import time 
#import numpy as np
import transforms3d as t3d

from autograd import grad, jacobian
import autograd.numpy as np
from autograd.numpy import linalg as LA
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

# pass in 1 channel of the img(x_dim,y_dim,1)
# return a matrix containing lat and long of each pixel (x, y, 2)
def find_long_lat(img):
    res_lat = 45/img.shape[0]
    res_long = 60/img.shape[1]
    cx = np.rint(img.shape[0]/2)
    cy = np.rint(img.shape[1]/2)
    
    long = np.zeros((img.shape[0], img.shape[1]))
    lat = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
       for j in range(img.shape[1]):
           long[i, j] = (cx - img[i,j]) * res_long
           lat[i, j] = (img[i,j] - cy) * res_lat
    return np.dstack((long, lat)) #stack long on top of lat

# converts spherical to cartesian coords
# input is (m, n) stacked long and lat matrix and depth(r)
# returns a matrix with x y z stacked depth wise (m, n, 3)
def sphr2cart(long_lat, r):
    long = long_lat[:,:,0]
    lat = long_lat[:,:,1]
    x = r*np.multiply(np.sin(lat), np.cos(long))
    y = r*np.multiply(np.sin(lat), np.sin(long))
    z = r*np.cos(lat)
    temp = np.dstack((x,y))
    return np.dstack((temp, z))

# converts cartesian to spherical coords
# input is matrix of cart coords x y z stacked depth wise
def cart2sphr(cart_mat):
    r = LA.norm(cart_mat, axis=2)
    long = np.arctan(LA.norm(cart_mat[:,:,:2], axis=2)/cart_mat[:,:,2])
    lat = np.arctan(cart_mat[:,:,1]/cart_mat[:,:,0])
    out = np.dstack((r, long))
    return np.dstack((out, lat))

# input: R is (3,3,N) collection of rot. matrices,
# cart_mat is a x mat, y mat, z mat stacked depth wise
# this function rotates each (x,y,z) coordinate pair of each 
# img by the respective rotation
# returns the rotate cart_mat points for EACH image
# so it would be a large stack of cart_mat for each image
def rotate(R, cart_mat, num_img):
    temp_lis = []
    vec_mat = cart_mat.reshape((cart_mat.shape[0]*cart_mat.shape[1]), cart_mat.shape[2]) #turns the 3d mat into a 2d one for rotation
    vec_mat = vec_mat.transpose()
    
    N = np.minimum(R.shape[2], num_img) # iterate for which ever there is less of
    for i in range(N):
        temp = np.matmul(R[:,:,i],vec_mat)
        temp = temp.transpose()
        temp = temp.reshape((cart_mat.shape[0],cart_mat.shape[1],3)) ##LOOK OVER THIS
        #print(temp.shape) # prints (3, 240, 320)
        temp_lis.append(temp)
            
    rot_stack = np.stack(temp_lis)
    return rot_stack
    # this stack basically contains the pos of 
    # each img in the sphere in cartesian coords

def cassini(spher):
    long = spher[:,:,:,1]
    lat = spher[:,:,:,2]
    
    x = np.zeros((spher.shape[0], spher.shape[1], spher.shape[2]))
    y = np.zeros((spher.shape[0], spher.shape[1], spher.shape[2]))
    for i in range(spher.shape[0]):
        x[i,:,:] = np.arcsin( np.cos(lat[i,:,:]) * np.sin(long[i,:,:]) )
        y[i,:,:] = np.arctan2( np.sin(lat[i,:,:]), np.cos(lat[i,:,:])*np.cos(long[i,:,:]) )
        
    temp = []
    temp.append(x)
    temp.append(y)
    out = np.stack(temp)
    
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


#imu_vals = imud.get("vals")
#imu_ts = imud.get("ts")

vic_mats = vicd.get("rots")
vic_ts = vicd.get("ts")

cam_im = camd.get("cam")
cam_ts = camd.get("ts")

#print(cam_im.shape[0])
#long_lat = np.zeros((cam_im.shape[0], cam_im.shape[1], 2))
R = vic_mats


test_img = cam_im[:,:,0,0]
lola = find_long_lat(test_img) # obtaining lat and long in spher. coords
img_cart = sphr2cart(lola, 1)
cart_stack = rotate(R, img_cart, cam_im.shape[3])

sphr_stack = np.zeros(cart_stack.shape)

# converts the cartesian coords to spherical
for i in range(cart_stack.shape[0]):
    sphr_stack[i,:,:,:] = cart2sphr(cart_stack[i,:,:,:])

print("done1")    

# using Cassini projection
temp = cassini(sphr_stack)
print(temp.shape)
        
print("done2")



'''
# Test to reshape a 3,3,N matrix into a (3*3), N matrix
x = np.arange(25).reshape((5,5))
y = np.arange(1,26).reshape((5,5))
z = np.arange(25).reshape((5,5))
temp = np.dstack((x,y))
out = np.dstack((temp,z))
out1 = out.reshape((out.shape[0]*out.shape[1]), out.shape[2])
out1 = out1.transpose() # turns it into 2d array so we can miltiply R by each point


#out2 = out1.reshape((3,5,5)) # this turns it back into original shape
out1 = out1.transpose()
out2 = out1.reshape((5,5,3)) # this turns it back into original shape

#out3 = np.stack((out1,out2), axis=3)
#out3 = np.stack((out3,out2), axis=3)
out3 = []
out3.append(out2)
out3.append(out2)
out3.append(out2)
out3.append(out2)
out4 = np.stack(out3)
out4 = out4.reshape((5,5,4,3))

'''