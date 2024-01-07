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
#include "C:\Users\po78\Documents\cuda_medkha\CUDA_TP\1_question\new_start\src\sys\time1.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

static long num_steps = 100000000, num_steps_per_thread = 1, num_threads_per_block = 1;
double step;

/*int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;  
    double thread_result = 0.0;
    double x; 
    for (int j = 0; j < num_steps_per_thread ; j++){
	if((i * num_steps_per_thread + j) < num_steps) {
		x = (i * num_steps_per_thread  + j - 0.5) * step_c;
		thread_result += 4.0/(1.0+x*x); 	
	}
    }
    atomicAdd(d_result, thread_result);
*/

__global__ 
void step_func(double* d_result, double step_c, long num_steps, long num_steps_per_thread) {
	// shared s data per block
    extern __shared__ double  s_data[]; 
    double x = 0.0;
    unsigned int tid = threadIdx.x;  
    unsigned int i = tid + blockIdx.x * blockDim.x;

    if( i < num_steps ) {
	    s_data[tid]=0;
	    for (int j = 0; j < num_steps_per_thread ; j++){
		if((i * num_steps_per_thread + j) < num_steps) {
			x = (i * num_steps_per_thread  + j - 0.5) * step_c;
			s_data[tid] += 4.0/(1.0+x*x); 	
		}
	    }
	    __syncthreads(); 
	    for(int s = 1; s < blockDim.x; s *= 2){
		int index = 2 * tid * s ; 
		if(index < blockDim.x) {
		    s_data[index] += s_data[index + s];
		}
		__syncthreads();
	    } 
	    if(tid == 0){
		d_result[blockIdx.x] = s_data[0];
	    } 
    }
}
/*	  for (i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
		  sum = sum + 4.0/(1.0+x*x);
	  }

	  pi = step * sum;
*/

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps_per_thread" ) == 0 ) ) {
            num_steps_per_thread = atol( argv[ ++i ] );
            printf( "  User num_stpes_per_thread is %ld\n", num_steps_per_thread );
        } else if ( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-num_threads" ) == 0 ) ) {
            num_threads_per_block =  atol( argv[ ++i ] );
            printf( "  User num_threads is %ld\n", num_threads_per_block );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	  double pi;
	  long double sum = 0.0;

	  double *h_result;
          size_t  n_blocks;	  
	    
	
      step = 1.0/(double) num_steps;

      // Timer products.
      struct timeval begin, end;

      gettimeofday( &begin, NULL );

// ***************** START of the part to parallelize with CUDA.
// we start with a block of 1 thread, which means we will have the number of blocks equals the number of steps.
// but there is a limit to the number of blocks that can be used in a grid. on x 2^31 -1  
    

// init device variables 

	double *d_result; 
	n_blocks = ceil((double)num_steps/(double)(num_threads_per_block * num_steps_per_thread)); 

	h_result = (double *)malloc(sizeof(double) * n_blocks);
	cudaMalloc((void **) &d_result, n_blocks * sizeof(double));


	/// cpy host variables into the device variables 
	cudaMemcpy(d_result, h_result, n_blocks* sizeof(double), cudaMemcpyHostToDevice); 
	// cpy the device variable to the host variable 
	// 2048 per sm 
	// TODO i would say the optimal is to lesser the blocks i think, as there is no use of the shared mem.
	step_func<<<n_blocks, num_threads_per_block,  num_threads_per_block * sizeof(double) >>>(d_result, step, num_steps, num_steps_per_thread); 
	cudaMemcpy(h_result, d_result,  n_blocks * sizeof(double), cudaMemcpyDeviceToHost); 
	for (int i = 0 ; i < n_blocks; i++ ) {
		sum += h_result[i];
	}
	// free the device variables 
	pi = step * sum;
	// ***************** END of the part to parallelize with CUDA.
	      
	      gettimeofday( &end, NULL );
	      // Calculate time.
	      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
			1.0e-6 * ( end.tv_usec - begin.tv_usec );
			
	      printf("\n pi with %ld steps, step= %lf, result = %lf , is %lf in %lf seconds\n ",num_steps,step,*h_result,pi,time);
	cudaFree(d_result);
	free(h_result);
}

