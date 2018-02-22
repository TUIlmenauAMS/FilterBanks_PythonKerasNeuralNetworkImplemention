# Low Delay Filter Bank Implementation using Python Keras 

Simple programs to use a keras convolutional neural network as implementation of a Low Delay filter bank. For literature about it, see:
https://www.idmt.fraunhofer.de/content/dam/idmt/documents/IL/Personal%20Websites/Schuller/publications/tsp8-96.pdf
and
https://www.idmt.fraunhofer.de/content/dam/idmt/documents/IL/Personal%20Websites/Schuller/publications/tsp3-00.pdf
and our lecture Multirate Signal Processing, 
https://www.tu-ilmenau.de/mt/lehrveranstaltungen/lehre-fuer-master-mt/multirate-signal-processing/
slides lecture 15.

Low Delay filter banks have lower system delay (over analysis and synthesis filter bank) than MDCT filter banks with comparable filters, or have better filters for the same system delay.

The advantages of a Keras implemenation are: 
* The possibility to use the GPU in this way and speed up the computation, 
* hence it might not need a specialized fast implementation,
* It can be easily integrated in Keras programs, for instance for source separation, or as a starting point for further optimization or training.
Gerald Schuller, November 2017.

## Getting Started
In the "main" section of the LDFB programs, the variable N=1024 is the number of subbands of the Low Delay Filter Bank. The function LDFBana_init reads the prototype filter coefficients from the file "h4096t2047d1024bbitc.txt".
It produces filter of length 4N.
The analysis program keras_LDFBanalysis.py reads in a sound file, for instance the "test.wav" file. In Keras the different subbands appear in its "channels" dimension.
The program writes the subband signals into the file "LDFB_subbands.pickle".
The synthesis program keras\_LDFBsynthesis.py reads in the file "LDFB\_subbands.pickle" and play back the reconstructed sound, using pyaudio.
The program keras\_LDFBanasyn.py calls the analysis and synthesis programs in a sequence.

To start, simply execute:
python keras\_LDFBanasyn.py



