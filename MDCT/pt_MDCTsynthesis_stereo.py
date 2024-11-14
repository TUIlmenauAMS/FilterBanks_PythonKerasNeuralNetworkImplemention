# MDCT synthesis filter bank using PyTorch Transpose Conv2D

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pickle
from sound import sound
import os
import sys


def MDCT_syn_init(shape):
    print("Initializing analysis MDCT weights, shape=", shape)
    N = shape[0]  # Number of subbands / outputs
    filtlen = shape[2] # kernel height
    nCh = shape[1]
    weights = np.zeros(shape)
    for ch in range(nCh):
        for k in range(N): # each subband / filter
            weights[k,ch,:,0] = (
                    np.sin(np.pi / filtlen * (np.arange(filtlen) + 0.5)) *
                    np.sqrt(2.0 / N) *
                    np.cos(np.pi / N * (k + 0.5) * 
                    (np.arange(filtlen) + 0.5 - N / 2)))
    print("Weights shape ", weights.shape) # Weight shape [out_channels, in_channels, kernel_height, kernel_width]

    return torch.tensor(weights, dtype=torch.float32)

class MDCTSynStereo(nn.Module):
    def __init__(self, N, filtlen):
        super(MDCTSynStereo, self).__init__()
        # Transposed convolution layer for synthesis
        self.deconv = nn.ConvTranspose2d(
            in_channels=N,   # Number of subbands
            out_channels=1,  # Reconstructed input
            kernel_size=(filtlen, 1),
            stride=(N, 1),
            padding=0,  # No padding for synthesis
            bias=False
        )
        # Initialize weights as per MDCT synthesis
        self.deconv.weight.data = MDCT_syn_init((N, 1, filtlen, 1)) # Weight shape [in_channels, out_channels, kernel_height, kernel_width]

    def forward(self, x):
        return self.deconv(x)

def pytorch_MDCT_syn_stereo(subbands, model):
    # Convert to a PyTorch tensor if it's still a NumPy array
    if isinstance(subbands, np.ndarray):
        subbands = torch.tensor(subbands, dtype=torch.float32)
    
    # expected input shape = [batch_size=1, in_channels=N, height=589, width=2]
    subbands = subbands.permute(2, 0, 1).unsqueeze(0)
    print("Transposed subbands shape =", subbands.shape)
    
    xrek = model(subbands).detach().numpy()
    xrek=xrek[0,0,:,0]
    print("Reconstructed shape =", xrek.shape)
    return xrek

if __name__ == '__main__':
    N = 1024  # Number of subbands/filters, stride
    filtlen = 2048  # Length of filter impulse response

    model = MDCTSynStereo(N, filtlen)
    with open("mdct_subbands_stereo.pickle", 'rb') as subfile:
        subbands = pickle.load(subfile)

    xrek = pytorch_MDCT_syn_stereo(subbands, model)
    os.system('espeak -ven -s 120 "The output of the stereo synthesis MDCT"')
    sound(2**15*xrek,44100)