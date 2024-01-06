
/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <algorithm>

#define __NUM_STEPS__ 1000 
static long num_steps = 100000000;
static long num_blocks = __NUM_STEPS__; 
double *step = (double*)malloc(sizeof(double));
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
// change the passing by value for the step
__global__ void reduce_without_s_mem(double *g_idata, double *g_odata, unsigned int n, const double step, unsigned int calculate_flag = 1) {    
    unsigned int tid = threadIdx.x;      
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(calculate_flag) {
	if(i < n) {
	    double x = ((double)i-0.5) * (step);
	    g_idata[i] = (double)4.0/(double)(1.0+x*x); 
	    //g_idata[i] = 1;
	}
	__syncthreads(); 
    }

    for(unsigned int s = 1; s < blockDim.x; s*=2) {
	int index =  2 * s * tid; 
	// as i am not copying the input data into shared memory
	// i had to change the index and also the condition.
	if(index <  blockDim.x && (blockIdx.x * blockDim.x + index + s < n)) {
	   g_idata[blockIdx.x * blockDim.x +  index] += g_idata[blockIdx.x * blockDim.x + index + s];   
	} 
	// before going to the next level we should have all the summations done.
	__syncthreads();
    } 
	if(tid == 0) {
	    g_odata[blockIdx.x] = g_idata[blockIdx.x * blockDim.x];
    } 
}
int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        }
	else if (strcmp( argv[ i ], "-B" ) == 0) {
	    num_blocks = atol( argv[++i] ); 
	    long num_blocks_min = (( num_steps + num_blocks - 1) / 1024) + 1;  
	    if(num_blocks < num_blocks_min) {
		num_blocks = num_blocks_min; 
		printf("Number of blocks is too small, the number of blocks is set to the minimum possible %ld. \n", num_blocks);
	    } 
	    // (__NUM_STEPS__ + num_blocks -1 ) / num_blocks;
	}
	 else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
// *************** CUDA 
 
// TODO free the h_step_mult

      // Timer products.
    
 // DECLARATION AND INITIALIZATION
    double *h_block_result = (double*)malloc(num_blocks * sizeof(double)); 
    double *h_result = (double*) malloc(2 * sizeof(double)); 
    double *d_steps, *d_result, *d_block_result; 
    long *d_size, *h_num_steps = &num_steps;
    double *h_steps = (double *)malloc( __NUM_STEPS__ * sizeof(double)); 


    *step = 1.0/(double) num_steps;
    std::fill_n(h_steps,__NUM_STEPS__, 0); 
    std::fill_n(h_block_result, num_blocks, 0);
    printf("the value of the first element of the array before : %lf. \n", h_steps[0]); 

    // CUDA ALLOC.
    cudaMalloc((void **)&d_steps,  __NUM_STEPS__ * sizeof(double)); 
    cudaMalloc((void **)&d_block_result, num_blocks * sizeof(double)); 
    cudaMalloc((void **)&d_size, sizeof(long)); 
    cudaMalloc((void **)&d_result, 2 * sizeof(double)); 
    printf("number of steps %ld.\n", *h_num_steps); 
    // CUDA CPY
    cudaMemcpy(h_num_steps, d_size, sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(h_steps, d_steps, __NUM_STEPS__ * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(h_block_result, d_block_result, num_blocks * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(h_result, d_result, 2 * sizeof(double), cudaMemcpyHostToDevice); 
    

// KERNELS 
    // Start time
    struct timeval begin, end;
    gettimeofday( &begin, NULL );

     //reduce_without_s_mem
    long threads = (num_steps +  num_blocks -1 ) / num_blocks; 
    reduce_without_s_mem<<<num_blocks, threads>>>(d_steps, d_block_result, num_steps, *step, 1); 
    cudaDeviceSynchronize();
    cudaMemcpy(h_block_result, d_block_result, num_blocks *  sizeof(double), cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_steps, d_steps, num_steps *  sizeof(double), cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
    reduce_without_s_mem<<<1, num_blocks>>>(d_block_result, d_result, num_blocks, *step, 0); 
    cudaDeviceSynchronize();
    cudaMemcpy(h_block_result, d_block_result, num_blocks *  sizeof(double), cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_result, d_result, 2  *  sizeof(double), cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
   

    // End time 
    gettimeofday( &end, NULL );

    // pi = step * sum 
    printf("the result before the multiplication with the step is %lf \n.", h_result[0]);
    h_result[0] = (1.0/num_steps) * (h_result[0]);
    printf("the first element of the array is : %lf \n", h_steps[0]); 
    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,h_result[0],time);

// CLEANING    
    cudaFree(d_size); 
    cudaFree(d_result); 
    cudaFree(d_steps); 
    cudaFree(d_block_result);
    free(h_result);
    free(h_block_result);

}	
/*  for (i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
		  sum = sum + 4.0/(1.0+x*x);
	  }

	  pi = step * sum;
*/

