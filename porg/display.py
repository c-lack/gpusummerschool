import matplotlib.pyplot as plt
import numpy as np

image = np.load('image.npy')

plt.imshow(image,cmap='gray')
plt.show()
