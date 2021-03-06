#MDCT analysis and synthesis filter bank, one after the other, for time measurement
#Gerald Schuller, November 2017

from keras_MDCTanalysis_stereo import *
from keras_MDCTsynthesis_stereo import *
from sound import sound
import time

N=1024 #Number of filters, stride
filtlen=2048 #Length of filter impulse response
modelana = generate_model_ana_stereo(N,filtlen)     # Compile an neural net analysis filter bank
modelsyn = generate_model_syn_stereo(N,filtlen)     # " Synthesis filter bank
fs, X= wav.read('teststereo.wav')
print("fs=", fs)
X=X*1.0/2**15
startime=time.time()
#Analysis Filter Bank:
Y=keras_MDCT_ana_stereo(X,modelana) 
#Synthesis Filter Bank:
xrek= keras_MDCT_syn_stereo(Y,modelsyn)
endtime=time.time()
print("Duration analysis-synthesis: ", endtime-startime)
os.system('espeak -ven -s 120 '+'"The output of the synthesis MDCT"')
sound(2**15*xrek,fs)


