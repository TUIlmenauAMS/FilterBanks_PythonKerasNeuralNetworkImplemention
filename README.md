# Filter Bank Implementations using Python Keras 

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

For details and examples of filter bank design, look at:
https://github.com/TUIlmenauAMS/Jupyter_notebooks_AMS/tree/master/Audio_Coding/Lec7_PQMF

and:

https://www.tu-ilmenau.de/mt/lehrveranstaltungen/lehre-fuer-master-mt/audio-coding/

Gerald Schuller, May 2020.
