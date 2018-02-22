#Filter Bank Implementations using Python Keras 

Keras implementations of Low Delay filter banks, Modified Discrete Cosine Transform filter banks, and Pseudo Quadrature Mirror filter banks

The advantages of the Keras implementation are: 
* The possibility to use the GPU in this way and speed up the computation, 
* hence it might not need a specialized fast implementation,
* It can be easily integrated in Keras programs, for instance for source separation, or as a starting point for further optimization or training.

It needs the Keras and Theano libraries, install it in Python2 with:
sudo pip install keras
sudo pip install Theano

in Python3:
sudo pip3 install keras
sudo pip3 install Theano

Gerald Schuller, February 2018.
