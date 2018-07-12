import numpy as np
import matplotlib.pyplot as plt

image = np.load('image.npy')

# Range over which we want to explore membership of the set
rmin = -2.0
rmax = 0.5
imin = -1.1
imax = 1.1

# Display the image
plt.figure(figsize = [9, 9])
plt.imshow(image,cmap='RdBu',extent=[rmin, rmax, imin, imax])
plt.show()
