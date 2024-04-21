from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


image = Image.open(f'/home/wutong/visual-tactile/VisualTactileData/tactile/1/1.png')
image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255.
print(image.shape)
image = image[:, image[0]<1].reshape(3,-1)
mean = image.mean(dim=-1)
var = image.var(dim=-1)
print(mean)
print(var)