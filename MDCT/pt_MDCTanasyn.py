# MDCT analysis and synthesis filter bank, one after the other, for time measurement
# Gerald Schuller, November 2017 (converted to PyTorch)

from pt_MDCTanalysis import torch_MDCT_ana, MDCTAnalysisNet
from pt_MDCTsynthesis import torch_MDCT_syn, MDCTSynthesisNet
from sound import sound
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

N = 1024  # Number of filters, stride
filtlen = 2048  # Length of filter impulse response

# Compile neural net models for analysis and synthesis filter banks
modelana = MDCTAnalysisNet(N, filtlen).to('cpu')  # Move models to CPU for simplicity
modelsyn = MDCTSynthesisNet(N, filtlen).to('cpu')

# Test Perfect Reconstruction and Delay:
X = np.arange(20 * N)
X_tensor = torch.tensor(X, dtype=torch.float32).to('cpu')  # Convert input to tensor

# MDCT Analysis:
start_time = time.time()
Y = torch_MDCT_ana(X_tensor, modelana)
analysis_time = time.time()

# MDCT Synthesis:
xrek = torch_MDCT_syn(Y, modelsyn)
synthesis_time = time.time()

# Print timings
print("Analysis time:", analysis_time - start_time)
print("Synthesis time:", synthesis_time - analysis_time)

# Convert reconstructed tensor back to numpy array for plotting
if isinstance(xrek, torch.Tensor):
    xrek_numpy = xrek.cpu().detach().numpy()  # Convert tensor to NumPy array
else:
    xrek_numpy = xrek  # If it's already a NumPy array, just assign it directly

# Plot original and reconstructed signals
plt.plot(X, label='Original')
plt.plot(xrek_numpy, label='Reconstructed')
plt.legend()
plt.title("Original and Reconstructed Test Signal")
plt.show()

# Now let's try with an actual audio file
fs, X_audio = wav.read('test.wav')
print("fs=", fs)

# Normalize audio signal
X_audio = X_audio * 1.0 / 2**15

# Convert to tensor
X_audio_tensor = torch.tensor(X_audio, dtype=torch.float32).to('cpu')

# Perform MDCT Analysis and Synthesis on the audio file
startime = time.time()

# Analysis Filter Bank
Y_audio = torch_MDCT_ana(X_audio_tensor, modelana)

# Synthesis Filter Bank
xrek_audio = torch_MDCT_syn(Y_audio, modelsyn)
endtime = time.time()

# Print total duration for analysis-synthesis
print("Duration analysis-synthesis: ", endtime - startime)

# Play the reconstructed audio (convert back to numpy and rescale)
if isinstance(xrek_audio, torch.Tensor):
    xrek_audio_numpy = xrek_audio.cpu().detach().numpy()  # Convert tensor to NumPy array
else:
    xrek_audio_numpy = xrek_audio  # If it's already a NumPy array, just assign it directly
sound(2**15 * xrek_audio_numpy, fs)
