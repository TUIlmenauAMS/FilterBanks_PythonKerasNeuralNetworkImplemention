# Filter Bank Implementations using Convolutional Layers with Python Keras and PyTorch

Keras and PyTorch implementations of Low Delay filter banks, Modified Discrete Cosine Transform filter banks, and Pseudo Quadrature Mirror filter banks

The advantages of the neural network implementation are: 
* The possibility to use the GPU in this way and speed up the computation, 
* hence it might not need a specialized fast implementation,
* It can be easily integrated in AI-based programs, for instance for source separation, or as a starting point for further optimization or training.

It needs the Keras, Theano and PyTorch libraries, install it in Python2 with:
sudo pip install keras
sudo pip install Theano
sudo pip install torch

in Python3:
sudo pip3 install keras
sudo pip3 install Theano
sudo pip3 install torch

For details and examples of filter bank design, look at:
https://github.com/TUIlmenauAMS/Jupyter_notebooks_AMS/tree/master/Audio_Coding

and:

https://www.tu-ilmenau.de/mt/lehrveranstaltungen/lehre-fuer-master-mt/audio-coding/

Gerald Schuller, May 2020.
