from PIL import Image        # Import the Python Image Library
import pyculib.fft as cufft  # Import the cuFFT library interface
from timeit import default_timer as timer  # Timer
import numpy as np
from numba import cuda

@cuda.jit
def multiply_elements(a, b, c):
    """
    Element-wise multiplication of a and b stored in c.
    """

    # What elements of a,b and c should this thread operate on?
    tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ty = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Better make sure the indices tx adn ty are inside the array!
    if ty < a.shape[0] and tx < a.shape[1]:
        c[ty, tx] = a[ty, tx] * b[ty, tx]

# Open an image file and convert to single colour (greyscale)
img = Image.open('porg.jpg').convert('L')
img_data = np.asarray(img,dtype=float)
dim = img_data.shape[0]

# Define the Gaussian to volume with
width = 0.2
domain = np.linspace(-5, 5,dim)
gauss = np.exp(-0.5*domain**2/(width*width))
shift = int(dim/2)
gauss = np.roll(gauss,shift)
gauss2D = gauss[:,np.newaxis] * gauss[np.newaxis,:]

# Make the data complex
img_data_complex = img_data + 1j * np.zeros((dim,dim))
gauss2D_complex = gauss2D + 1j * np.zeros((dim,dim))

# Arrays to store intermediate result and final output on host
img_fft = np.empty((dim,dim),dtype=complex)
gauss_fft = np.empty((dim,dim),dtype=complex)
img_ifft = np.empty((dim,dim),dtype=complex)

# Put the data on the device
d_img_data_complex = cuda.to_device(img_data_complex)
d_gauss2D_complex = cuda.to_device(gauss2D_complex)

# Create device arrays
d_img_fft = cuda.device_array((dim,dim),dtype=np.complex)
d_gauss_fft = cuda.device_array((dim,dim),dtype=np.complex)
d_img_ifft = cuda.device_array((dim,dim),dtype=np.complex)

t1 = timer()

# FFT the two input arrays
cufft.fft(d_img_data_complex,d_img_fft)
cufft.fft(d_gauss2D_complex,d_gauss_fft)

# Copy data back to host
img_fft=d_img_fft.copy_to_host()
gauss_fft = d_gauss_fft.copy_to_host()

# Multiply each element in fft_img by the corresponding image in fft_gaus
img_conv = img_fft * gauss_fft

# Copy to the device
d_img_conv = cuda.to_device(img_conv)

# Inverse Fourier transform
cufft.ifft(d_img_conv,d_img_ifft)

# Copy result back to host
img_ifft = d_img_ifft.copy_to_host()

t2 = timer()

# Elapsed time (in milliseconds)
print("Convolution with multiplication on host took : ",1000*(t2-t1)," milliseconds.")

t1 = timer()  # Start timer

# FFT the two input arrays
cufft.fft(d_img_data_complex, d_img_fft)
cufft.fft(d_gauss2D_complex, d_gauss_fft)

# Use the kernel to multiply on the device
threads_per_block = 32
blocks_per_grid = dim // threads_per_block + 1

multiply_elements[blocks_per_grid, threads_per_block](d_img_fft, d_gauss_fft, d_img_conv)

# Inverse Fourier transform
cufft.ifft(d_img_conv, d_img_ifft)

# Copy result back to host
img_ifft = d_img_ifft.copy_to_host()

t2 = timer()

np.save('image.npy',img_ifft.real)

# Elapsed time (in milliseconds)
print("Convolution with multiplication on device took : ",1000*(t2-t1)," milliseconds.")
