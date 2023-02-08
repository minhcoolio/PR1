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

# pass in 1 channel of the img(x_dim,y_dim,1)
# return a matrix containing lat and long of each pixel
def find_lat(img):
    res_lat = 45/img.shape[0]
    res_long = 60/img.shape[1]
    
    cx = np.rint(img.shape[0]/2)
    cy = np.rint(img.shape[1]/2)
    
    long = (cx - img) * res_long
    lat = (img - cy) * res_lat
def find_long():
    print(1)

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

print(cam_im.shape[0])