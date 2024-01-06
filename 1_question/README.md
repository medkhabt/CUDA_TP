# Max threads per block: 
I had a problem with the N blocks parallization when i have small block numbers, as i do calculate threads from the number of block and size of the matrix that i need to calculate. 

By cloning the project from nvidia `cuda-samples` , speciffically the project `device query` I found out that the max threads per block is 1024
I will put the log of the program in the same folder for context . ( file name is : "device_query.log" )  
So for now i will cap the number of threads per block.


# atomicAdd for doubles 

AtomicAdd is not available for double for the version i am using, i had to use the implementation that i found in the nvidia website for it. but didn't really read it for now.
## Using atomicAdd instead of normal summation    
For now I only have one single shared mem that i write to. So i can't just add elems to it. 

# thread with many iterations

I am certainly doing something wrong. I divided L by the end to get the result but it was just a fix. The problem hsould be in the kernel function ( index ? , number of threads ) Also when i increase the number L it give inaccurate results 

> Interrupted. Continue this afterwards. 

# Reduction lvl2 

## Working with the global memory 

each thread is responsible only for one element of the array. 

SO the max we can calculate is the 1024[threads/block] * 32[block/grid?]

for the precision it gives us result with error of 0.01 at best.

## result depends on the threads count

@_TODO_ i should probably check the accuracy of the integral calculation based on the diffirent number of threads used for calculating the integral. As sum association is not associative when it comes to floating points. 

## Workign with the shared memory 

In case i choose not a big iteration number it works fine, and outputs a result that has an error of +/- 0.01 but when i try for example iteration number of 16384 with 32 blocks which mean while using half capacity per block (512 thread) i get a cuda failure when trying to run the kernel `cudaLaunchKernel` , for now i still don't know the reason of this problem. it may have a relation with the max register count.  
# Info about the integral calculation 
I had some inaccurate results when using cuda. So i spent some little time to understand the process. i found out that the way we calculate the integral is by takign the value of the middle element of each block we created from splitting the graph surface. And calculating a rectangle surface for that block. and summing all the blocks. 

# Order of args matter
I just found out that i need to write the -b arg after defining the number of iterations. I need to fix this dependency but it 's not really the most important thing to do right now .

# the multi stage reduction 

I should probably ask the prof about this, but from my understanding. it is required from us to use multiple grids. 


# Cuda-gdb 

## cuda api failures 

To make the gdb stop when a cuda api failure happens, you must activate it by `set cuda api_failures stop` before running the code. 


and after that you type `where` to see the stack of the exception.

The problem was that the max memory that can be allocated in the shared mem for a block is less than what i am trying to allocate. 


the max size per block is (49152b)
