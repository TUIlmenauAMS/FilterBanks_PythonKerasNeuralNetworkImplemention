# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a keras convolutional neural network as implementation of an PQMF synthesis filter bank.
The advantage is the possibility to use the GPU in this way and speed up the computation.
Gerald Schuller, November 2017.
"""

from keras.models import Sequential
#from keras.layers import LSTM, Lambda
from keras.layers.core import Dense, Activation
#from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
#from keras.layers import GaussianNoise
#from keras.constraints import unit_norm
#from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
#import theano.tensor as tht
#from keras_correlation_loss import corr_loss
#import keras.backend
import os
import sys

if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

"""
Input Dimension:
https://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape
(nb_of_examples, nb_of_features, 1)
In 2D:
Input x dimension: (batch size, height, width,channels (e.g. RGB or stereo))
Input to the layers without the batch size.
"""


def PQMF_init(shape, dtype=None):
    print("Initializing MDCT weights")
    N= shape[-1] #Number of subbands
    filtlen=shape[0]
    weights=np.zeros(shape)
    #For N=1024 subbands:
    #qmfwin=np.loadtxt("qmf1024_8x.mat")
    #mirror the other half:
    #qmfwin=np.append(qmfwin,np.flipud(qmfwin))
    #for N=64 subbands:
    qmfwin=np.loadtxt('qmf.dat');
    print("qmfwin.shape=",qmfwin.shape) 
    for k in range(N):
        weights[:,0,0,k]= qmfwin * np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(np.arange(filtlen)+0.5-N/2))
        #print("weights[:,0,0,k]=", weights[:,0,0,k])
    #print("test orthogonality:", np.dot(np.transpose(weights[:,0,0,:]),weights[:,0,0,:]))
    return weights


def generate_model_syn(N,filtlen):
    #    Method to construct a fully connected neural network using keras and theano.
    #    :return: Trainable object
    # Define the model. Can be sequential or graph
    #For the autoencoder
    model = Sequential()

    #The addition of the synthesis subbands comes from choosing filters=1! This is one (2D) multiband filter, 
    #which fits its input, hence applied to all subbands at once, and added up in the end. 
    #The weights are indeed a matrix. 
    model.add(Conv2DTranspose(filters=1, kernel_size=(filtlen,1), strides=(N,1), activation="linear", use_bias=False, kernel_initializer=PQMF_init, input_shape=(None,1,N)))
    
    #Output shape: 4D tensor with shape:
    #(batch, new_rows, new_cols, filters) if data_format='channels_last'. 

    # Compile appropriate theano functions
    #losses: https://keras.io/losses/
    #mean_squared_error ('mse'), mean_absolute_error(y_true, y_pred), mean_squared_logarithmic_error,...
    #model.compile(loss='mean_absolute_error', optimizer='sgd')
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def keras_PQMF_syn(subbands,model):
    """MDCT Synthesis Filter bank implemented with Keras.
       argument: Y: a 2D array containing the subbands, the last dim. is the subband index
       returns: xrek, 1D array of the input (audio) signal
    """
    #Make the dimensionality suitable for keras:
    subbands=np.expand_dims(subbands,axis=0)
    subbands=np.expand_dims(subbands,axis=2)
    print("subbands.shape=", subbands.shape)
    xrek=model.predict(subbands) # Compute the synthesis MDCT
    print("xrek.shape=", xrek.shape)
    #Extract the right dimension for the reconstructed audio sognal:
    xrek=xrek[0,:,0,0]
    return xrek

if __name__ == '__main__':
    from sound import sound
    #N=1024 #Number of filters, stride
    #filtlen=8192 #Length of filter impulse response
    N=64
    filtlen=640
    model = generate_model_syn(N,filtlen) 
    with open("pqmf_subbands.pickle", 'rb') as subfile:
       subbands=pickle.load(subfile)
    xrek= keras_PQMF_syn(subbands,model)
    os.system('espeak -ven -s 120 '+'"The output of the synthesis PQMF"')
    sound(2**15*xrek,16000)
    
