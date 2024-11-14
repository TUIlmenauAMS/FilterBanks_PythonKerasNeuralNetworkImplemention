import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pickle

# Initialize the MDCT analysis filter weights
def MDCT_ana_init(filtlen, N):
    print("Initializing MDCT analysis weights")
    weights = np.zeros((N, 1, filtlen, 1), dtype=np.float32)
    for k in range(N):
        weights[k, 0, :, 0] = np.sin(np.pi/filtlen*(np.arange(filtlen)+0.5)) * np.sqrt(2.0/N) * np.cos(np.pi/N*(k+0.5)*(np.arange(filtlen)+0.5-N/2))
    return torch.from_numpy(weights)

# Define the MDCT analysis filter bank model using PyTorch
class MDCTAnalysisNet(nn.Module):
    def __init__(self, N, filtlen):
        super(MDCTAnalysisNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=N, kernel_size=(filtlen, 1), stride=(N, 1), bias=False)
        self.conv.weight.data = MDCT_ana_init(filtlen, N)

    def forward(self, x):
        return self.conv(x)

# Function to perform MDCT analysis
def torch_MDCT_ana(X, model):
    """MDCT Analysis Filter bank implemented with PyTorch.
       argument: X: 1D array of the input (audio) signal
       returns: Y, a 2D array containing the subbands, the last dim. is the subband index
    """
    # Check if X is a NumPy array before converting
    if isinstance(X, np.ndarray):
    # If X is a NumPy array, convert it to a PyTorch tensor
        Xp = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)  # Shape: [batch_size, channels, height, width]
    else:
    # If X is already a tensor, you can directly reshape it if needed
        Xp = X.float().unsqueeze(0).unsqueeze(1).unsqueeze(3)  # Ensure X is a float tensor and reshape  # Shape: [batch_size, channels, height, width]
    with torch.no_grad():  # Disable gradient computation for inference
        subbands = model(Xp)  # Shape: [batch_size, filters, new_rows, 1]
    
    # Squeeze the last dimension and reorder the tensor
    subbands = subbands.squeeze(-1).permute(0, 2, 1)  # Shape: [batch_size, new_rows, filters]
    
    Y = subbands[0].numpy()  # Convert to NumPy (since batch_size is 1, we take the first item)
    return Y

if __name__ == '__main__':
    N = 1024  # Number of filters, stride
    filtlen = 2 * N  # Length of filter impulse response

    model = MDCTAnalysisNet(N, filtlen)  # Initialize the model

    # Load an audio file (replace 'test.wav' with your file)
    fs, X = wav.read('test.wav')
    print("fs=", fs)
    X = X * 1.0 / 2**15  # Normalize the audio data

    # Perform MDCT analysis
    Y = torch_MDCT_ana(X, model)

    # Plot the MDCT spectrogram
    plt.imshow(Y.T, aspect='auto')  # Transpose Y for correct display (filters on y-axis)
    plt.title('The MDCT Filter Bank Spectrogram')
    plt.xlabel('Block Number')
    plt.ylabel('Subband Index')
    plt.show()

    # Save the MDCT subbands to a file
    with open("mdct_subbands.pickle", 'wb') as subfile:
        pickle.dump(Y, subfile)
