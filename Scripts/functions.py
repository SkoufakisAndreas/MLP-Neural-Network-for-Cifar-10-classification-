#function opens cifar files assigns them into an obj fo
#then from fo stores everything to batch
#Finaly stores at x the data and y the label matrixes and returns them

import numpy as np
import torch

def load_batch(file):
     import pickle

     with open(file, 'rb' ) as fo:
         batch = pickle.load(fo, encoding= 'bytes')

     #Extract and convert to float 32 the elements to be able to do calculations later
     x = batch[b'data'].astype(np.float32)    #Contains array of images 10000x3072
     y = np.array(batch[b'labels'], dtype = np.int64)    #Contains array of labels 10000x1

     return x, y

#This function converts the images to grayscale reducing columns from 3072 to 1024
#Images are stored as [RED(1024) | GREEN(1024) | BLUE(1024) ]
def grayscale(array):

     R = array[:, :1024]
     G = array[:, 1024:2048]
     B = array[:, 2048:3072]

     gray = (R + G + B)/3

     return gray

#This function uses a 2*2 block and replace its with its mean
#That way array size is 4 times smaller

def downsampling(array):
     #Create the downsampled array
     down_array = np.empty((array.shape[0], 768), dtype = np.float32)

     j = 0
     for i in range(768):
          down_array[:, i] = (array[:,j] + array[:, j+1]
                              + array[:, 32 + j] + array[:, 33 + j]) /4
          j = j + 2

          #When completing 2 full rows jump to the third
          if(i + 1) % 16 == 0:
               j+=32

     return down_array

# Finds the direction of greatest variance in the training data.
# Returns its weights and the mean vector for centering.
def pca(array):

     #Create the mean values array
     mean_array = array.mean(dim=0)
     #Center the array
     x_centered = array - mean_array
     N = x_centered.size(0) -1
     #Create covariance array
     C = (x_centered.T @ x_centered)/ N
     #Extract eigenvalues and vectors in ascending order of eigenvalue
     _, eigenvectors = torch.linalg.eigh(C)
     #Keep last 128 columns and flip the column dimension so to get
     #The 128 eigenvectors corresponding to the 128 largest eigenvalues
     w = eigenvectors[:, -128:].flip(dims = (1, ))
     #return μ, w to construct the same way both train and test data

     return w, mean_array


