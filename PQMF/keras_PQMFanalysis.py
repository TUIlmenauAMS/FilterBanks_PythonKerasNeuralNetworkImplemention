# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'Gerald Schuller'
__copyright__ = 'G.S.'

"""
Simple program to use a keras convolutional neural network as implementation of an QMF analysis filter bank.
The advantage is the possibility to use the GPU in this way and speed up the computation.
Gerald Schuller, November 2017.
"""

from keras.models import Sequential
#from keras.layers import LSTM, Lambda
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import Conv2DTranspose
#from keras.constraints import unit_norm
#from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
#import theano.tensor as tht
#from keras_correlation_loss import corr_loss
import keras.backend
from sound import sound
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
    #for 1024 subbands:
    #qmfwin=np.loadtxt("qmf1024_8x.mat")
    #mirror the other half:
    #qmfwin=np.append(qmfwin,np.flipud(qmfwin))
    #for 64 subbands:
    qmfwin=np.loadtxt('qmf.dat');
    print("qmfwin.shape=",qmfwin.shape) 
    for k in range(N):
        weights[:,0,0,k]= qmfwin * np.sqrt(2.0/N)*np.cos(np.pi/N*(k+0.5)*(np.arange(filtlen)+0.5-N/2))
        #print("weights[:,0,0,k]=", weights[:,0,0,k])
    #print("test orthogonality:", np.dot(np.transpose(weights[:,0,0,:]),weights[:,0,0,:]))
    return weights

def generate_model_ana(N, filtlen):
    model = Sequential()
   
    #The filter coefficients are the weights of the convolutional layer:
    #input_shape=(siglen,1,1)
    model.add(Conv2D(filters=N, kernel_size=(filtlen,1), strides=(N,1), activation="linear", use_bias=False, kernel_initializer=PQMF_init, input_shape=(None,1,1)) )


    #Output shape: 4D tensor with shape:
    #(batch, new_rows, new_cols, filters) if data_format='channels_last'. 

    # Compile appropriate theano functions
    #losses: https://keras.io/losses/
    #mean_squared_error ('mse'), mean_absolute_error(y_true, y_pred), mean_squared_logarithmic_error,...
    #model.compile(loss='mean_absolute_error', optimizer='sgd')
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def keras_PQMF_ana(X,model):
    """MDCT Analysis Filter bank implemented with Keras.
       argument: X: 1D array of the input (audio) signal
       returns: Y, a 2D array containing the subbands, the last dim. is the subband index
    """
    Xp=np.expand_dims(X,axis=0)
    Xp=np.expand_dims(Xp,axis=2)
    Xp = np.expand_dims(Xp, axis=-1) #Last dimension: channels, like stereo (here: mono)
    subbands=model.predict(Xp) # Process the signal with the autoencoder
    #print("subbands.shape=", subbands.shape)
    #subbands.shape= (1, 44713, 1, 4)
    Y=subbands[0,:,0,:]
    return Y 

#Testing:
if __name__ == '__main__':
    #N=1024 #Number of filters, stride
    #filtlen=8192 #Length of filter impulse response
    N=64
    filtlen=640

    model = generate_model_ana(N,filtlen)     # Compile an neural net
    fs, X= wav.read('test.wav')
    print("fs=", fs)
    X=X*1.0/2**15
    #Make the dimensionality suitable for Keras:
    Y=keras_PQMF_ana(X,model) 
    plt.imshow(Y.T)
    plt.show()
    subfile=open("pqmf_subbands.pickle", 'wb')
    #dump the right dimensions for the subbands, last dimension: subband index:
    pickle.dump(Y ,subfile)

