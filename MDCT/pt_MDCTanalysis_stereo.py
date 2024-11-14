import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pickle

def MDCT_ana_init(shape):
    print("Initializing analysis MDCT weights, shape=", shape)
    N = shape[0]  # Number of subbands / outputs
    filtlen = shape[2] # kernel height
    nCh = shape[1] # number of input channels
    weights = np.zeros(shape)
    for ch in range(nCh): # each input channel (width)
        for k in range(N): # each subband / filter
            weights[k,ch,:,0] = (
                    np.sin(np.pi / filtlen * (np.arange(filtlen) + 0.5)) *
                    np.sqrt(2.0 / N) *
                    np.cos(np.pi / N * (k + 0.5) * 
                    (np.arange(filtlen) + 0.5 - N / 2)))
    print("Weights shape ", weights.shape) # Weight shape [out_channels, in_channels, kernel_height, kernel_width]

    return torch.tensor(weights, dtype=torch.float32)

class MDCTAnaStereo(nn.Module):
    def __init__(self, N, filtlen):
        super(MDCTAnaStereo, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=N,
            kernel_size=(filtlen, 1),
            stride=(N, 1),
            padding=(0, 0),
            bias=False
        )
        
        self.conv.weight.data = MDCT_ana_init((N, 1, filtlen, 1)) # Weight shape [out_channels, in_channels, kernel_height, kernel_width]

    def forward(self, x):
        return self.conv(x)

def pytorch_MDCT_ana_stereo(X, model):
    """MDCT Analysis Filter bank implemented with PyTorch.
       argument: X: 2D array of the input (audio) signal
       returns: Y, a 2D array containing the subbands, the last dim. is the subband index
    """
    # Convert X to a PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    X = X.unsqueeze(0).unsqueeze(0)
    # print("X input tensor shape", X.shape)
    print(model)
    # expected input shape = [batch_size=1, in_channels=2, height=604160, width=1]
    subbands = model(X)  # Process the signal with the model
    # print("output subbands shape=", subbands.shape)  # Check the shape after processing
    Y = subbands.squeeze(0).permute(1, 2, 0)  # Remove the batch dimension and reorder dimensions
    print("Y shape", Y.shape)
    return Y

if __name__ == '__main__':
    N = 1024  # Number of filters
    filtlen = 2048  # Length of filter impulse response

    model = MDCTAnaStereo(N, filtlen)  # Instantiate the model
    fs, X = wav.read('teststereo.wav')
    print("fs=", fs)
    X = X * 1.0 / 2**15  # Normalize audio data
    print("X.shape=", X.shape)

    # Perform MDCT analysis
    Y = pytorch_MDCT_ana_stereo(X, model).detach().numpy()
    print("Y.shape=", Y.shape)

    # Plot the results for left and right channels
    plt.imshow(Y[:, 0, :].T)  # Left channel
    plt.title('Left Channel')
    plt.figure()
    plt.imshow(Y[:, 1, :].T)  # Right channel
    plt.title('Right Channel')
    plt.show()

    # Save the subbands
    with open("mdct_subbands_stereo.pickle", 'wb') as subfile:
        pickle.dump(Y, subfile)  # Convert to NumPy array before saving
