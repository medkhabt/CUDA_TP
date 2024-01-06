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

static long num_steps = 100000000;
double step;
int num_blocks = 1;

__global__ 
void step_func(double* d_result, double step_c, long num_steps) {
    int i  = blockIdx.x;  
    double block_result = 0.0;
    double x; 
    int block_iter_size = (int) ceil((double) num_steps / (double) gridDim.x);
    for (int j = 0; j < block_iter_size ; j++){
	if(i * block_iter_size + j >= num_steps)
	    break;
	x = (i * block_iter_size + j - 0.5) * step_c;
	block_result += 4.0/(1.0+x*x); 	
    }
    __syncthreads();
    printf("block result for %d is %lf, and the size is %ld \n", i, block_result, (int)ceil((double) num_steps / (double) gridDim.x));
    atomicAdd(d_result, block_result); 
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
        }
	if ( ( strcmp( argv[ i ], "-B" ) == 0 ) || ( strcmp( argv[ i ], "-num_blocks" ) == 0 ) ) {
            num_blocks = atol( argv[ ++i ] );
            printf( "  User num_block is %ld\n", num_blocks);
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	  double pi;

	  double *h_result;
	  
      step = 1.0/(double) num_steps;
      if(num_blocks > num_steps){
	num_blocks = num_steps; 
	printf("num_blocks specified are greater than num_steps, num_blocks are set to num_steps.");
      }
      // Timer products.
      struct timeval begin, end;

      gettimeofday( &begin, NULL );

// ***************** START of the part to parallelize with CUDA.
// we start with a block of 1 thread, which means we will have the number of blocks equals the number of steps.
// but there is a limit to the number of blocks that can be used in a grid. on x 2^31 -1  
    

// init device variables 

double *d_block, *d_result; 
cudaMalloc((void **) &d_result, sizeof(double));
cudaMalloc((void **) &d_block, num_steps * sizeof(double));

// cpy host variables into the device variables 
//cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice); 
// cpy the device variable to the host variable 

step_func<<<num_blocks,1>>>(d_result, step, num_steps); 
cudaMemcpy(h_result, d_result,  sizeof(double), cudaMemcpyDeviceToHost); 

// free the device variables 
pi = step * (*h_result);
// ***************** END of the part to parallelize with CUDA.
      
cudaFree(d_result);
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps, step= %lf, result = %lf , is %lf in %lf seconds\n ",num_steps,step,*h_result,pi,time);
}

