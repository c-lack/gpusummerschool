import numpy as np                 # Numpy
from numba import cuda, float64    # Cuda and numba float64 datatype

from timeit import default_timer as timer  # Timer

@cuda.jit
def sum_row(d_a, d_sum):
    """Given a device array a, calculate the sum over elements in each row."""

    # Get row and column of current element
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    column = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Create shared memory array to use for summation
    sum_sh = cuda.shared.array(3, dtype=float64)

    if (row < d_a.shape[0]):
        if column == 0 :
            sum_sh[row] = 0.0 # First thread in row initialises sum

    cuda.syncthreads()

    if (row < d_a.shape[0]):
        if column < d_a.shape[1]:

            # Add to element 'row' of array sum_sh. Note that we
            # don't need to read from global memory anymore
            cuda.atomic.add(sum_sh, row , d_a[row, column])

    cuda.syncthreads()

    # Write result to global memory
    if (row < d_a.shape[0]):
        if column == 0 : d_sum[row] = sum_sh[row]

# Generate random 3 x 2 array
my_array = np.random.rand(32, 32)

t1 = timer()
my_sum_cpu = np.sum(my_array, axis=1)
t2 = timer()

print("cpu in %f milliseconds" % (1000*(t2-t1)))
print(my_sum_cpu)

# Copy data to device and create new array for output
d_my_array = cuda.to_device(my_array)
d_my_sum   = cuda.device_array(32, dtype=np.float64)

# Launch a single thread block of 2 x 3 threads
t1 = timer()
sum_row[(1, 1), (32, 32)](d_my_array, d_my_sum)
t2 = timer()

# Copy result back and print it
my_sum_gpu = d_my_sum.copy_to_host()
print("gpu in %f milliseconds" %(1000*(t2-t1)))
print(my_sum_gpu)
