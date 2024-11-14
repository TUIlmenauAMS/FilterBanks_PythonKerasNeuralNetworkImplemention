# MDCT analysis and synthesis filter bank, one after the other, for time measurement

from pt_MDCTanalysis_stereo import *
from pt_MDCTsynthesis_stereo import *
from sound import sound
import time

N=1024 # Number of subbands/filters, stride
filtlen=2048 # Length of filter impulse response
modelana = MDCTAnaStereo(N, filtlen)     # Analysis filter bank NN model
modelsyn = MDCTSynStereo(N, filtlen)     # Synthesis filter bank NN model
fs, X= wav.read('teststereo.wav')        # Sampling rate
print("fs=", fs)
X=X*1.0/2**15
startime=time.time()
Y=pytorch_MDCT_ana_stereo(X,modelana)    # Analysis filter bank 
xrek= pytorch_MDCT_syn_stereo(Y,modelsyn)# Synthesis filter bank
endtime=time.time()
print("Duration analysis-synthesis: ", endtime-startime)
os.system('espeak -ven -s 120 '+'"The output of the synthesis MDCT"')
sound(2**15*xrek,fs)


