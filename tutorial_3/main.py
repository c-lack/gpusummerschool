import numpy as np                 # Numpy
from numba import cuda, float64    # Cuda and numba float64 datatype

from timeit import default_timer as timer  # Timer

@cuda.jit
def diffusion_kernel(D, invdx2, invdy2, dt, d_u, d_u_new):
    """
    Simple kernel to evolve a function U forward in time according to an explicit FTCS
    finite difference scheme. Arguments are...

    D       : Diffusion coefficient
    invdx2  : 1/(dx^2) where dx is the grid spacing in the x direction
    invdy2  : 1/(dy^2) where dy is the grid spacing in the y direction
    dt      : time step
    d_u     : Device array storing U at the current time step
    d_u_new : Device array storing U at the next time step
    """

    # Which row and column on the simulation grid should this thread use
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Check that the index lies inside the grid
    if row < d_u.shape[0] and col < d_u.shape[1]:

        # Neighbour cells using period boundary conditions
        up   = (row + 1)%d_u.shape[0]
        down = (row - 1)%d_u.shape[0]

        left  = (col - 1)%d_u.shape[1]
        right = (col + 1)%d_u.shape[1]

        # Compute second derivatives of u w.r.t. x and y
        d2udx2 = (d_u[row,left]  - 2.0*d_u[row,col] + d_u[row,right])*invdx2
        d2udy2 = (d_u[down,col]  - 2.0*d_u[row,col] + d_u[up,col])*invdy2

        # Populate u_new with the time-evolved function
        d_u_new[row, col] = d_u[row, col] + D * dt * ( d2udx2 + d2udy2 )

@cuda.jit
def diffusion_kernel_shared(D, invdx2, invdy2, dt, d_u, d_u_new):
    """
    Kernel to evolve a function U forward in time according to an explicit FTCS
    finite difference scheme. Shared memory is used. Arguments are...

    D       : Diffusion coefficient
    invdx2  : 1/(dx^2) where dx is the grid spacing in the x direction
    invdy2  : 1/(dy^2) where dy is the grid spacing in the y direction
    dt      : time step
    d_u     : Device array storing U at the current time step
    d_u_new : Device array storing U at the next time step
    """

    # Shared array large enough to store 30x30 block + "halo" of boundary neighbours
    # N.B. the size of the shared array is set at compile time, but see also
    # https://stackoverflow.com/questions/30510580/numba-cuda-shared-memory-size-at-runtime
    u_sh = cuda.shared.array((32, 32), dtype=float64)

    # Row and column in global matrix - 32 x 32 grid including halo
    row = cuda.threadIdx.y + cuda.blockIdx.y * ( cuda.blockDim.y - 2 ) - 1
    col = cuda.threadIdx.x + cuda.blockIdx.x * ( cuda.blockDim.x - 2 ) - 1

    # Row and column in shared memory tile
    sh_row = cuda.threadIdx.y
    sh_col = cuda.threadIdx.x

    # Apply periodic boundary conditions
    row = row%d_u.shape[0]
    col = col%d_u.shape[1]

    # Copy from device memory to shared memory
    u_sh[sh_row, sh_col] = d_u[row, col]

    # Do not proceed until all threads reach this point
    cuda.syncthreads()

    # Only threads which belong to the interior 30 x 30 grid compute
    # (The other 32^2 - 30^30 = 124 threads do nothing)
    if sh_row > 0 and sh_row < 31 and sh_col > 0 and sh_col < 31:

        left  = sh_col - 1
        right = sh_col + 1

        up   = sh_row + 1
        down = sh_row - 1

        # Compute second derivatives of u w.r.t. x and y
        d2udx2 = (u_sh[sh_row, left]  - 2.0*u_sh[sh_row, sh_col] + u_sh[sh_row, right])*invdx2
        d2udy2 = (u_sh[down, sh_col]  - 2.0*u_sh[sh_row, sh_col] + u_sh[up, sh_col])*invdy2

        # Populate u_new with the time-evolved function
        d_u_new[row, col] = u_sh[sh_row, sh_col] + D * dt * ( d2udx2 + d2udy2)

    u_sh = None

# Create an empty array
dim = 960
c = np.zeros((dim, dim))

# Fill the middle of the grid with a concentration of 1.0
for irow in range(c.shape[0] // 4, 3*c.shape[0] // 4):
    for icol in range(c.shape[1] // 4, 3*c.shape[1] // 4):
        c[irow, icol] = 1.0

# We want this to represent a square domain spanning 0 -> 1 in each direction
domain = [0.0, 1.0, 0.0, 1.0]

D = 1.0  # Diffusion coefficient

x_spacing = 1.0/float(c.shape[0])
y_spacing = 1.0/float(c.shape[1])

# Store spacing as inverse square to avoid repeated division
inv_xspsq = 1.0/(x_spacing**2)
inv_yspsq = 1.0/(y_spacing**2)

# Satisfy stability condition
time_step = 0.25*min(x_spacing,y_spacing)**2

# Copy this array to the device, and create a new device array to hold updated value
d_c = cuda.to_device(c)
d_c_new = cuda.device_array(c.shape, dtype=np.float)

threads_per_block = (32, 32)
blocks_per_grid   = (dim//30, dim//30)

t1 = timer()  # Start timer

# Evolve forward 2000 steps
for step in range(2000):

    # Launch the kernel
    diffusion_kernel_shared[blocks_per_grid,threads_per_block](D, inv_xspsq, inv_yspsq, time_step, d_c, d_c_new)

    # Swap the identit of the old and new device arrays
    d_c, d_c_new = d_c_new, d_c

cuda.synchronize()  # make sure all threads finish before stopping timer

t2 = timer()
print("Simulation using simple kernel took : ",1000*(t2-t1)," milliseconds.")

# Copy the current concentration from the device to the host
c = d_c.copy_to_host()

np.save('image.npy',c)
