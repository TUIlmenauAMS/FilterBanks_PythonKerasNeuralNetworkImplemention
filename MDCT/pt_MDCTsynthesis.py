import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sound import sound

# Initialize the MDCT synthesis filter weights
def MDCT_syn_init(filtlen, N):
    print("Initializing MDCT synthesis weights")
    weights = np.zeros((N, 1, filtlen, 1), dtype=np.float32)
    for k in range(N):
        weights[k, 0, :, 0] = np.sin(np.pi / filtlen * (np.arange(filtlen) + 0.5)) * np.sqrt(2.0 / N) * np.cos(np.pi / N * (k + 0.5) * (np.arange(filtlen) + 0.5 - N / 2))
    return torch.from_numpy(weights)

# Define the MDCT synthesis filter bank model using PyTorch
class MDCTSynthesisNet(nn.Module):
    def __init__(self, N, filtlen):
        super(MDCTSynthesisNet, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=N, out_channels=1, kernel_size=(filtlen, 1), stride=(N, 1), bias=False)
        self.deconv.weight.data = MDCT_syn_init(filtlen, N)

    def forward(self, x):
        return self.deconv(x)

# Function to perform MDCT synthesis
def torch_MDCT_syn(subbands, model):
    """MDCT Synthesis Filter bank implemented with PyTorch."""
    # Prepare the input tensor for PyTorch
    subbands = torch.from_numpy(subbands).float()
    # print("subbands shape before reshape=", subbands.shape)

    # Reshape to match the expected input shape: [batch_size, in_channels, height, width]
    subbands = subbands.unsqueeze(0).unsqueeze(2).permute(0, 3, 1, 2)  # Shape: [1, N (subbands), blocks, 1]
    print("subbands shape ", subbands.shape)
    
    with torch.no_grad():  # Disable gradient computation for inference
        xrek = model(subbands)  # Compute the synthesis MDCT
        print("xrek.shape=", xrek.shape)
    
    # Extract the relevant dimensions for the reconstructed audio signal
    xrek = xrek.squeeze().numpy()  # Flatten to 1D array
    return xrek

if __name__ == '__main__':
    N = 1024  # Number of filters, stride
    filtlen = 2 * N  # Length of filter impulse response
    fs = 16000  # Sampling rate (assumed)

    # Initialize the MDCT synthesis model
    model = MDCTSynthesisNet(N, filtlen)

    # Load the MDCT subbands from the pickle file
    with open("mdct_subbands.pickle", 'rb') as subfile:
        subbands = pickle.load(subfile)

    # Perform MDCT synthesis
    xrek = torch_MDCT_syn(subbands, model)

    # Rescale and ensure mono audio playback
    if len(xrek.shape) == 1:  # Mono audio
        print("Playing mono audio")
        sound(2**15 * xrek, fs)
    else:
        print("Error: Unexpected audio shape.")
