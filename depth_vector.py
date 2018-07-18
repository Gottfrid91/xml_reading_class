import tensorflow as tf
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

def get_depth_vector(img):
    def find_nearest_min(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def find_nearest_max(array, value):
        idx = (np.abs(array - value)).argmax()
        return idx

    def get_zero_patches(idx_zero):

        def split(arr, cond):
            return [arr[cond], arr[~cond]]

        diff_vector = np.zeros(1+idx_zero.shape[0])
        diff_vector[1:]= idx_zero
        differences = np.subtract(idx_zero, diff_vector[0:-1])[1:]
        indices = np.where(differences>1)
        #print(indices, differences)
        zero_patches = []

        if indices[0].size != 0:
            #extract first zero patch
            zero_patches.append(idx_zero[np.where(idx_zero <= idx_zero[indices[0][0]])])
            #extract the following zero patches
            for i in range(indices[0][:].shape[0],0,-1):
                zero_patches.append(idx_zero[np.where(idx_zero > idx_zero[indices[0][i-1]])])
        if indices[0].size == 0:
            zero_patches.append(idx_zero)
        return(zero_patches)

    depth_vector = np.zeros(img.shape[1])
    for i in range(0,img.shape[1]):
        layer = np.argwhere(img[:,i])
        if layer.size != 0:
            depth_vector[i] = max(layer) - min(layer)
    #interpolation of missing deoth information
    #get all indices
    idx_nonzero = np.argwhere(depth_vector)
    idx_zero = np.where(depth_vector==0)[0]
    #check if list is empty = no zero patches
    if len(idx_zero) != 0:
        #get list with seperate zero patches
        zero_patches = get_zero_patches(idx_zero)
        #print(zero_patches)
        #find interpolation value
        for patch in zero_patches:
            #print(patch)
            closest_min = find_nearest_min(idx_nonzero, min(patch))
            closest_max = find_nearest_min(idx_nonzero, max(patch))
            #print(closest_min, closest_max)
            #print(depth_vector[idx_nonzero[closest_min]],depth_vector[idx_nonzero[closest_max]])
            interpolation = (depth_vector[idx_nonzero[closest_min]]+depth_vector[idx_nonzero[closest_max]])/2
            #print(interpolation)
            #interpolate
            depth_vector[patch] = interpolation

    return depth_vector
