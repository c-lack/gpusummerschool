import numpy as np
from numba import cuda

def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

# Create the device function mandel_gpu from the function "mandel" above
mandel_gpu = cuda.jit(device=True)(mandel)

@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):

    # Get the dimensions of the grid from the image device array
    dimx = image.shape[1]
    dimy = image.shape[0]

    # Work out spacing between elements
    pixel_size_x = (max_x - min_x) / dimx
    pixel_size_y = (max_y - min_y) / dimy

    # What elements of the image should this thread operate on?
    tx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ty = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Coordinates in the complex plane
    real = min_x + tx * pixel_size_x
    imag = min_y + ty * pixel_size_y

    # Count number of interations needed to diverge
    if ty < dimy and tx < dimx:
        image[ty, tx] = mandel_gpu(real, imag, iters)

# Array to hold the output image - i.e. number of iterations
# as an unsigned 8 bit integer
image = np.zeros((1000, 1500), dtype = np.uint8)

# Range over which we want to explore membership of the set
rmin = -2.0 ; rmax = 0.5
imin = -1.1 ; imax = 1.1

# Maximum number of iterations before deciding "does not diverge"
maxits = 20

# The image size above is chosen to map onto a whole number of threadblocks.
# IMPORTANT - we normally think of arrays indexed as row, column hence y, x
# The tuples specifiying the thread grid dimensions are indexed as x, y
threads_per_block = (32, 32)

bx = image.shape[1] // threads_per_block[1] + 1
by = image.shape[0] // threads_per_block[0] + 1

blocks_per_grid = (bx, by)

# Copy image to a device array which we will populate in our kernel
d_image = cuda.to_device(image)

# Launch the kernel, passing the range of x and y to use
mandel_kernel[blocks_per_grid, threads_per_block](rmin, rmax, imin, imax, d_image, maxits)

# Copy the resulting image back to the host
image = d_image.copy_to_host()

np.save('image.npy',image)
