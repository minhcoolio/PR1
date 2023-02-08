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
    
    long = (cx - img) * res_long
    lat = (img - cy) * res_lat
    
    long = np.zeros((img.shape[0], img.shape[1]))
    lat = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
       for j in range(img.shape[1]):
           long[i, j] = (cx - img[i,j]) * res_long
           lat[i, j] = (img[i,j] - cy) * res_lat
    return np.dstack((long, lat)) #stack long on top of lat

# converts spherical to cartesian coords
# input is (m, n) long and lat matrix and depth(r)
# returns a matrix with x y z stacked depth wise (m, n, 3)
def sphr2cart(long, lat, r):
    x = r*np.multiply(np.sin(lat), np.cos(long))
    y = r*np.multiply(np.sin(lat), np.sin(long))
    z = r*np.cos(lat)
    temp = np.dstack((x,y))
    return np.dstack((temp, z))

# input: R is (3,3,N) collection of rot. matrices,
#        cart_mat is a x mat, y mat, z mat stacked depth wise
# this function rotates each (x,y,z) coordinate pair of each 
# img by the respective rotation
def rotate(R, cart_mat):
    x = 1
    

dataset="1"
cfile = "../data/cam/cam" + dataset + ".p"
ifile = "../data/imu/imuRaw" + dataset + ".p"
vfile = "../data/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")


imu_vals = imud.get("vals")
imu_ts = imud.get("ts")

vic_mats = vicd.get("rots")
vic_ts = vicd.get("ts")

cam_im = camd.get("cam")
cam_ts = camd.get("ts")

#print(cam_im.shape[0])
#long_lat = np.zeros((cam_im.shape[0], cam_im.shape[1], 2))

# Test to reshape a 3,3,N matrix into a (3*3), N matrix
x = np.arange(25).reshape((5,5))
y = np.arange(25).reshape((5,5))
z = np.arange(25).reshape((5,5))
temp = np.dstack((x,y))
out = np.dstack((temp,z))
out1 = out.reshape((out.shape[0]*out.shape[1]), out.shape[2])
out1 = out1.transpose() # turns it into 2d array so we can miltiply R by each point

#out2 = out1.transpose()
out2 = out1.reshape((3,5,5)) # this turns it back into original shape

