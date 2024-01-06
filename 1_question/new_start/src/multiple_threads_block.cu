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
#include "sys/time.h"

static long num_steps = 100000000;
double step;
// TODO implement the step.
size_t n_threads = 1024, steps_per_thread = 1; 
__global__ 
void step_func(double* d_result, double *d_block, double step_c, long num_steps) {
    double x = 0.0;
    int tid = threadIdx.x;  
    int i = tid + blockIdx.x * blockDim.x;
    x = (i-0.5)*step_c;
    d_block[tid] = 4.0/(1.0+x*x); 
    __syncthreads(); 
    for(int s = 1; s < blockDim.x; s *= 2){
	int index = 2 * tid * s ; 
	if(index < blockDim.x) {
	    d_block[index] += d_block[index + s];
	}
	__syncthreads();
    } 
    if(tid == 0) d_result[blockIdx.x] = d_block[0];
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
        } else if ( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-num_threads" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	  double pi,sum = 0.0;

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

double *d_result, *d_block; 

n_blocks = ceil( num_steps/n_threads); 
printf("the number of blocks are %d \n", n_blocks); 

h_result = (double *)malloc(n_blocks);
cudaMalloc((void **) &d_result, n_blocks * sizeof(double));
cudaMalloc((void **) &d_block, n_threads* sizeof(double));


// cpy host variables into the device variables 
cudaMemcpy(d_result, h_result, n_blocks* sizeof(double), cudaMemcpyHostToDevice); 
// cpy the device variable to the host variable 
// 2048 per sm 
// TODO i would say the optimal is to lesser the blocks i think, as there is no use of the shared mem.
step_func<<<n_blocks, n_threads>>>(d_result, d_block, step, num_steps); 
cudaMemcpy(h_result, d_result,  n_blocks * sizeof(double), cudaMemcpyDeviceToHost); 
printf("\n*****************************\n");
for (int i = 0 ; i < n_blocks; i++ ) {
	sum += h_result[i];
}
// free the device variables 
pi = step * sum;
// ***************** END of the part to parallelize with CUDA.
      
cudaFree(d_result);
cudaFree(d_block);
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps, step= %lf, result = %lf , is %lf in %lf seconds\n ",num_steps,step,*h_result,pi,time);
free(h_result);
}

