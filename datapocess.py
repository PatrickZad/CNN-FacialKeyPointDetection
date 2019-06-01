import numpy as np
import torch
def trainingImage2tensor( imagestr):
        pixelstrlist = imagestr.split(' ')
        pixel_array = np.array([int(pixel) for pixel in pixelstrlist])
        return torch.from_numpy(pixel_array.reshape((96, 96)))
