#LDFB analysis and synthesis filter bank, one after the other, for time measurement
#Gerald Schuller, November 2017

from keras_LDFBanalysis import *
from keras_LDFBsynthesis import *
from sound import sound
import time
from scipy.signal import freqz

N=1024 #Number of filters, stride
filtlen=4096 #Length of filter impulse response
modelana = generate_model_ana(N,filtlen)     # Compile an neural net analysis filter bank
modelsyn = generate_model_syn(N,filtlen)     # " Synthesis filter bank

#Test Perfect reconstruction and Delay:
X=np.arange(20*N)

a=time.time()
Y=keras_LDFB_ana(X,modelana)
b=time.time()
xrek= keras_LDFB_syn(Y,modelsyn)
c=time.time()
print("Analysis time:", b-a)
print("Synthesis time:", c-b)
#Plots reconstructed ramp if ok:
#print("xrek.shape=", xrek.shape)
#print("xrek=", xrek)
plt.plot(-X)
plt.plot(xrek)
plt.legend(('Original', 'Reconstructed'))
plt.title("Original and Recontructed Test Signal")
plt.show()
#Delay is relative to the delay of an orthogonal filter bank of equal length! 
#Here: length=4096 taps, analysis-synthesis delay of an equal length orthogonal filter bank: 4095 samples.
#Low delay filter bank: 2047 samples delay, difference: -2048 samples
#Detailed comparison of the 2 ramp signals gives indeed exactly -2048!

#get and plot the synthesis impulse response of subband 0:
Y=np.ones((10,N))*0.0;
Y[0,0]=1.0
xrek= keras_LDFB_syn(Y,modelsyn)
plt.plot(xrek)
plt.title('Impulse Response of Subband 0 of the Synthesis Low Delay FB')
w,H=freqz(xrek,worN=4096)
plt.figure()
plt.plot(w,20*np.log10(np.abs(H)/np.max(np.abs(H))+1e-6))
plt.title('Its Magnitude Frequency Response') 
plt.ylabel('dB Attenuation')
plt.xlabel('Normalized Frequency (pi is Nyquist Freq.)')
plt.show()

fs, X= wav.read('test.wav')
print("fs=", fs)
print("X.shape=", X.shape)
X=X*1.0/2**15
startime=time.time()
#Analysis Filter Bank:
Y=keras_LDFB_ana(X,modelana) 
#Synthesis Filter Bank:
xrek= keras_LDFB_syn(Y,modelsyn)
print("xrek.shape=", xrek.shape) #output may be a little shorter, to make signal fit into an integer number of blocks of size N!
endtime=time.time()
print("Duration analysis-synthesis: ", endtime-startime)


#Delay estimation:
mse=np.zeros(3*N)
for n in range(3*N):
   #Mean squared error:
   minlen=min(len(X[n:]),len(xrek))
   #-X because of the sign change after reconstruction!
   mse[n]=np.mean((-X[n:(n+minlen)]-xrek[:minlen])**2)
print("Delay reconstructed to original=", np.argmin(mse), "np.min(mse)=",np.min(mse) ) #argmin=2048, min=4e-5
#Keras convolutional layers already compensate for a delay of the length of the filters!
#plt.plot(mse)
#plt.title("Mean Squared Error for different Delays")
plt.show()

os.system('espeak -ven -s 120 '+'"The output of the synthesis Low Delay Filter Bank"')
sound(2**15*xrek,fs)


